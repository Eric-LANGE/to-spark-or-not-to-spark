"""
PySpark ETL and ML Pipeline for Image Featurization and PCA
Chaîne de traitement distribué pour extraction de caractéristiques visuelles (features) et réduction de dimension (PCA)
"""

import os
import sys
import logging
import subprocess
import socket
from io import BytesIO
import boto3
from typing import Iterator, List, Optional

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from pyspark.ml.feature import PCA, StandardScaler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def _configure_io() -> logging.Logger:
    """
    Configure stdout to be line-buffered and route Python logging to stdout
    That complements PYTHONUNBUFFERED=1 set by the EMR step
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    logging.basicConfig(
        level=getattr(
            logging, os.environ.get("P9_LOG_LEVEL", "INFO").upper(), logging.INFO
        ),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for name in ("py4j", "py4j.clientserver", "py4j.java_gateway"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("p9")


def check_parquet_exists(spark: SparkSession, path: str) -> bool:
    """
    Checks if a parquet source dataset exists and is valid by checking for _SUCCESS file
    Uses the Hadoop FileSystem API configured for Spark
    """
    if not path.endswith("/"):
        path += "/"
    success_path = f"{path}_SUCCESS"
    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    URI = jvm.java.net.URI
    FileSystem = jvm.org.apache.hadoop.fs.FileSystem
    Path = jvm.org.apache.hadoop.fs.Path
    try:
        fs = FileSystem.get(URI.create(success_path), conf)
        return fs.exists(Path(success_path))
    except Exception:
        return False


def _build_config() -> dict:
    """
    Build and validate the configuration dictionary from environment variables
    """
    return {
        "s3_endpoint": os.environ.get("P9_S3_ENDPOINT", "s3.eu-west-3.amazonaws.com"),
        "bucket": os.environ.get("P9_S3_BUCKET", "p9.data"),
        "samples_per_category": int(os.environ.get("P9_SAMPLES_PER_CATEGORY", "45")),
        "pca_k": int(os.environ.get("P9_PCA_K", "128")),
        "target_files": os.environ.get("P9_TARGET_FILES", "").strip().lower(),
        "arrow_batch": int(os.environ.get("P9_ARROW_BATCH", "128")),
    }


def main():
    """Main function to orchestrate the image featurization and PCA pipeline"""
    logger = _configure_io()
    cfg = _build_config()

    # 0. REGION DETECTION
    s3_endpoint_str = cfg["s3_endpoint"]
    s3_region = "unknown"
    try:
        if "amazonaws.com" in s3_endpoint_str:
            parts = s3_endpoint_str.split(".")
            if len(parts) > 2 and parts[0] == "s3" and parts[1] != "amazonaws":
                s3_region = parts[1]  # e.g., 'eu-west-3'
            elif s3_endpoint_str == "s3.amazonaws.com":
                s3_region = "us-east-1"  # Default S3 legacy endpoint
    except Exception as e:
        logger.warning(
            f"Could not parse S3 region from endpoint '{s3_endpoint_str}': {e}"
        )

    emr_region = "unknown"
    try:
        session = boto3.Session()
        emr_region = session.region_name
        if emr_region is None:
            emr_region = os.environ.get(
                "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "unknown")
            )
    except ImportError:
        logger.warning("boto3 not found. Falling back to env vars for EMR region.")
        emr_region = os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "unknown")
        )
    except Exception as e:
        logger.warning(
            f"Could not determine EMR region via boto3: {e}. Falling back to env vars."
        )
        emr_region = os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "unknown")
        )

    # 1. SPARK SESSION INITIALIZATION
    s3_endpoint = cfg["s3_endpoint"]
    spark = (
        SparkSession.builder.appName("P9_EMR_PyTorch")
        .config("spark.sql.parquet.writeLegacyFormat", "true")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .getOrCreate()
    )
    spark.conf.set(
        "spark.sql.execution.arrow.maxRecordsPerBatch", str(cfg["arrow_batch"])
    )
    sc = spark.sparkContext

    try:
        # 2. PARAMETERS AND DATA LOADING
        bucket = cfg["bucket"]
        pca_k = cfg["pca_k"]
        samples_per_category = cfg["samples_per_category"]

        path_base = f"s3://{bucket}"
        path_data = f"{path_base}/Test"
        path_data_parquet = f"{path_base}/Test_Compacted.parquet"
        path_result = f"{path_base}/fruit_categories_pca"

        total_executor_cores = sc.defaultParallelism
        n_partitions = total_executor_cores * 2

        logger.info(f"Spark version: {spark.version}")
        logger.info(f"sc.defaultParallelism: {sc.defaultParallelism}")
        logger.info(f"Partitions: {n_partitions}")
        logger.info(f"Input path:    {path_data}")
        logger.info(f"Optimized path: {path_data_parquet}")
        logger.info(f"Output path:   {path_result}")
        logger.info(f"Arrow batch:   {cfg['arrow_batch']}")
        logger.info(f"S3 Region (from endpoint): {s3_region}")
        logger.info(f"EMR Cluster Region (detected): {emr_region}")

        # --- Data Loading Parquet-first ---
        if check_parquet_exists(spark, path_data_parquet):
            logger.info(f"Optimized path found. Loading from: {path_data_parquet}")
            images_df_raw = spark.read.parquet(path_data_parquet).cache()
        else:
            logger.warning(
                f"Optimized path not found. Performing first-time load from: {path_data}"
            )
            # Load all images
            images_df_raw = (
                spark.read.format("binaryFile")
                .option("pathGlobFilter", "*.jpg")
                .option("recursiveFileLookup", "true")
                .load(path_data)
                .withColumn("label", F.element_at(F.split(F.col("path"), "/"), -2))
                .cache()
            )

            # --- Write to Parquet for next time ---
            logger.info(
                f"Writing data to optimized Parquet format: {path_data_parquet}"
            )
            try:
                (
                    images_df_raw.select("path", "label", "content")
                    .write.mode("overwrite")
                    .parquet(path_data_parquet)
                )
                logger.info("Successfully wrote compacted Parquet for future runs.")
            except Exception as e:
                logger.warning(
                    f"Could not write compacted Parquet, will retry next time: {e}"
                )

        # Report folder with the fewest files
        min_row = (
            images_df_raw.groupBy("label")
            .agg(F.count("*").alias("count"))
            .orderBy(F.col("count").asc(), F.col("label").asc())
            .first()
        )
        if min_row:
            logger.info(
                f"Category with fewest files: '{min_row['label']}' ({min_row['count']} files)"
            )

        # Stratified sampling from the raw dataset
        windowSpec = Window.partitionBy("label").orderBy("path")
        images_df = (
            images_df_raw.withColumn("row_num", F.row_number().over(windowSpec))
            .filter(F.col("row_num") <= samples_per_category)
            .drop("row_num")
        )

        images_df.cache()
        total_images = images_df.count()
        category_count = images_df.select("label").distinct().count()
        logger.info(
            f"Loaded {total_images} images from {category_count} categories "
            f"({samples_per_category} per category)."
        )

        # --- GPU Probe on executors ---
        def _probe_partitions(_):
            host = socket.gethostname()
            try:
                import torch

                avail, cu = torch.cuda.is_available(), torch.version.cuda
            except Exception:
                avail, cu = False, None
            # nvidia-smi summary
            try:
                out = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=name",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    )
                    .strip()
                    .splitlines()
                )
                if out:
                    name = out[0].strip()
                    gpu_short = name.split()[-1] if name else "?"
                    line = f"host={host} | GPU={gpu_short} | cuda={avail} (cu={cu})"
                else:
                    line = f"host={host} | cuda={avail} (cu={cu}) | (no output)"
            except Exception as e:
                line = f"host={host} | cuda={avail} (cu={cu}) | nvidia-smi unavailable ({e})"
            yield line

        par = max(3, min(12, sc.defaultParallelism or 3))
        raw = sc.parallelize(range(par), par).mapPartitions(_probe_partitions).collect()
        seen = set()
        logger.info("--- GPU probe (executors) ---")
        for rec in raw:
            host_key = rec.split("|", 1)[0].strip()
            if host_key in seen:
                continue
            seen.add(host_key)
            logger.info(rec)

        # 3. MODEL PREPARATION AND FEATURIZATION
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        driver_model = mobilenet_v2(weights=weights)
        driver_model.classifier = torch.nn.Identity()
        driver_model.eval().to("cpu")
        bcast_state = sc.broadcast(
            {k: v.cpu() for k, v in driver_model.state_dict().items()}
        )
        del driver_model

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        _preprocess = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        # --- Apply Featurization ---
        FEAT_DIM = 1280  # MobileNetV2 penultimate layer size
        logger.info("--- Apply featurization ---")
        logger.info(
            f"Pandas UDF: MobileNetV2 → array<float>[{FEAT_DIM}] | partitions={n_partitions}"
        )

        def _to_features_aligned(
            bytes_list: List[bytes], model: torch.nn.Module, dev: str
        ) -> List[Optional[list]]:
            """
            Vectorize a batch of images while preserving alignment with input rows
            Returns a list of feature arrays or None (for invalid images)
            """
            valid_idx: List[int] = []
            tensor_list: List[torch.Tensor] = []
            for i, b in enumerate(bytes_list):
                try:
                    img = Image.open(BytesIO(b)).convert("RGB")
                    tensor_list.append(_preprocess(img))
                    valid_idx.append(i)
                except Exception as e:
                    logging.getLogger("p9").warning(
                        f"Invalid image skipped at batch index {i}: {e}"
                    )
            if tensor_list:
                batch = torch.stack(tensor_list, dim=0).to(dev, non_blocking=True)
                with torch.no_grad():
                    feats = model(batch)
                np_feats = feats.detach().cpu().numpy().astype("float32")
            else:
                np_feats = None
            out: List[Optional[list]] = [None] * len(bytes_list)
            if np_feats is not None:
                for j, i in enumerate(valid_idx):
                    out[i] = list(np_feats[j])
            return out

        @F.pandas_udf("array<float>")
        def featurize_udf(
            content_series_iter: Iterator[pd.Series],
        ) -> Iterator[pd.Series]:
            """
            Pandas iterator UDF that loads a MobileNetV2 per executor,
            produces 1280-dim float features per image and yields None for invalid images
            """
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_grad_enabled(False)
            model = mobilenet_v2(weights=None)
            model.classifier = torch.nn.Identity()
            model.load_state_dict(bcast_state.value, strict=True)
            model.eval().to(dev)

            for content_series in content_series_iter:
                aligned = _to_features_aligned(content_series.tolist(), model, dev)
                yield pd.Series(aligned)

        # Apply featurization
        features_df = images_df.repartition(n_partitions).select(
            F.col("path"),
            F.col("label"),
            featurize_udf(F.col("content")).alias("features"),
        )

        # --- Invalid image stats per category ---
        invalid_stats_df = (
            features_df.groupBy("label")
            .agg(
                F.count(F.lit(1)).alias("total"),
                F.sum(
                    F.when(F.col("features").isNull(), F.lit(1)).otherwise(F.lit(0))
                ).alias("invalid"),
            )
            .withColumn("invalid_rate", F.col("invalid") / F.col("total"))
        )
        # Global summary
        global_stats = (
            invalid_stats_df.agg(
                F.sum("total").alias("g_total"),
                F.sum("invalid").alias("g_invalid"),
            )
            .withColumn("g_invalid_rate", F.col("g_invalid") / F.col("g_total"))
            .collect()[0]
        )
        logger.info(
            f"Invalid images (global): {int(global_stats['g_invalid'])} / {int(global_stats['g_total'])} "
            f"({float(global_stats['g_invalid_rate'])*100:.2f}%)"
        )
        # Top categories by invalid_rate
        if int(global_stats["g_invalid"]) > 0:
            top_cats = (
                invalid_stats_df.orderBy(
                    F.col("invalid_rate").desc(), F.col("invalid").desc()
                )
                .limit(5)
                .toPandas()
            )
            if not top_cats.empty:
                logger.info(
                    "Top categories by invalid_rate (worst 5):\n"
                    + top_cats.to_string(index=False)
                )

        # 4. PCA AND SAVE
        # Remove invalid rows prior to scaling/PCA
        features_df = features_df.filter(F.col("features").isNotNull())
        df_vec = features_df.withColumn(
            "features_vec", array_to_vector("features")
        ).cache()

        # --- StandardScaler ---
        # Scale features (divide by std-dev) before PCA
        # This ensures features with high variance don't dominate the PCA
        logger.info("Applying StandardScaler (withStd=True, withMean=False)...")
        scaler = StandardScaler(
            inputCol="features_vec",
            outputCol="scaled_features",
            withStd=True,
            withMean=False,  # PCA will handle centering (withMean=True by default)
        )
        scaler_model = scaler.fit(df_vec)
        df_scaled = scaler_model.transform(df_vec)
        logger.info("StandardScaler applied.")

        logger.info(f"Running PCA with requested k={pca_k}...")
        # Robust PCA: cap k by feasible limits and fall back if necessary
        row_count_for_pca = df_scaled.count()
        FEAT_DIM = 1280
        max_k = (
            max(1, min(pca_k, FEAT_DIM, row_count_for_pca - 1))
            if row_count_for_pca > 1
            else 1
        )
        k_try = max_k
        pca_model = None
        last_err = None
        while k_try >= 1 and pca_model is None:
            try:
                pca = PCA(k=k_try, inputCol="scaled_features", outputCol="pca_features")
                pca_model = pca.fit(df_scaled)
            except Exception as e:
                last_err = e
                logger.warning(f"PCA(k={k_try}) failed, trying smaller k ... ({e})")
                k_try = k_try - max(1, min(16, k_try // 4))
        if pca_model is None:
            logger.error(f"PCA failed after fallback attempts. Last error: {last_err}")
            raise last_err
        effective_k = pca_model.getK()
        if effective_k != pca_k:
            logger.info(
                f"PCA k adjusted from {pca_k} → {effective_k} (feasible limit)."
            )
        pca_k = effective_k
        pca_df = pca_model.transform(df_scaled).select("path", "label", "pca_features")

        out_cols = ["path", "label"] + [
            vector_to_array("pca_features")[i].alias(f"pca_{i}") for i in range(pca_k)
        ]
        out_df = pca_df.select(*out_cols)

        # Write (coalesce optionnel)
        _tf_env = cfg["target_files"]
        if _tf_env and _tf_env != "auto":
            try:
                target_files = max(1, int(_tf_env))
                out_df.coalesce(target_files).write.mode("overwrite").parquet(
                    path_result
                )
                logger.info(
                    f"PCA with K={pca_k} completed. Exported to: {path_result} (coalesce={target_files})"
                )
            except Exception as _e:
                out_df.write.mode("overwrite").parquet(path_result)
                logger.warning(
                    f"PCA with K={pca_k} completed. Exported to: {path_result} (coalesce=off, reason={_e})"
                )
        else:
            out_df.write.mode("overwrite").parquet(path_result)
            logger.info(
                f"PCA with K={pca_k} completed. Exported to: {path_result} (coalesce=off)"
            )

        # 5. QUALITY CHECKS
        logger.info("--- Quality Checks ---")
        verified_df = spark.read.parquet(path_result)
        pca_cols = [f"pca_{i}" for i in range(pca_k)]

        count_after = verified_df.count()
        logger.info(f"Data integrity: {count_after} rows successfully written to disk.")

        nan_exprs = [
            F.sum(F.isnan(F.col(c)).cast("long")).alias(f"{c}_nan") for c in pca_cols
        ]
        sum_abs_cols = None
        for c in pca_cols:
            sum_abs_cols = (
                F.abs(F.col(c))
                if sum_abs_cols is None
                else (sum_abs_cols + F.abs(F.col(c)))
            )
        with_norm_df = verified_df.withColumn("sum_abs", sum_abs_cols)
        eps = F.lit(1e-12)
        agg_row = with_norm_df.agg(
            *nan_exprs,
            F.sum(F.when(F.col("sum_abs") <= eps, F.lit(1)).otherwise(F.lit(0))).alias(
                "zero_vector_rows"
            ),
        ).first()
        agg_results = agg_row.asDict()

        nan_cols = {k: v for k, v in agg_results.items() if k.endswith("_nan")}
        total_nans = int(sum(nan_cols.values()))
        logger.info(
            f"NaN Values: Found {total_nans} NaN values across all PCA columns."
        )
        if total_nans > 0:
            logger.warning({k: int(v) for k, v in nan_cols.items() if v > 0})

        zero_vector_count = int(agg_results["zero_vector_rows"])
        logger.info(
            f"Zero Vectors: Found {zero_vector_count} rows where all PCA components are zero."
        )

        explained_variance = pca_model.explainedVariance
        total_explained_variance = sum(explained_variance)
        logger.info("--- Explained variance ---")
        logger.info(
            f"Total variance explained by {pca_k} components: {total_explained_variance:.4f}"
        )
        for i, var in enumerate(explained_variance[:5]):
            logger.info(f"Component {i}: {var:.4f}")

        # --- Parquet output report ---
        try:
            files = verified_df.inputFiles()
            jvm = spark._jvm
            conf = spark._jsc.hadoopConfiguration()
            URI = jvm.java.net.URI
            FileSystem = jvm.org.apache.hadoop.fs.FileSystem
            Path = jvm.org.apache.hadoop.fs.Path
            fs = FileSystem.get(URI.create(path_result), conf)
            sizes = []
            for p in files:
                try:
                    sz = int(fs.getFileStatus(Path(p)).getLen())
                    sizes.append(sz)
                except Exception:
                    pass
            n = len(files)
            if n > 0 and sizes:
                total = sum(sizes) / (1024 * 1024)
                avg = (sum(sizes) / n) / (1024 * 1024)
                mn = min(sizes) / (1024 * 1024)
                mx = max(sizes) / (1024 * 1024)
                logger.info("--- Parquet Output report ---")
                logger.info(
                    f"Files: {n} | Total: {total:.2f} MiB | Avg: {avg:.2f} MiB | Min/Max: {mn:.2f}/{mx:.2f} MiB"
                )
        except Exception as e:
            logger.warning(f"[Parquet report skipped: {e}]")

        # PCA sample preview
        logger.info("--- PCA output (3 random samples) ---")
        num_pca_to_show = min(5, pca_k)
        display_cols = ["path", "label"] + [f"pca_{i}" for i in range(num_pca_to_show)]
        sample_df = verified_df.select(*display_cols)
        for i in range(num_pca_to_show):
            col_name = f"pca_{i}"
            sample_df = sample_df.withColumn(
                col_name, F.rpad(F.format_number(F.col(col_name), 4), 10, " ")
            )
        sample_df.orderBy(F.rand()).limit(3).show(truncate=False)

        logger.info("--- Spark Job completed successfully ---")

    finally:
        if "images_df_raw" in locals():
            images_df_raw.unpersist()
        if "images_df" in locals():
            images_df.unpersist()
        if "df_vec" in locals():
            df_vec.unpersist()
        if "df_scaled" in locals():
            df_scaled.unpersist()
        spark.stop()
        logging.getLogger("p9").info("--- Spark Session stopped ---")


if __name__ == "__main__":
    main()
