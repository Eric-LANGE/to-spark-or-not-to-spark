# Pipeline d'extraction de features visuelles et PCA sur AWS EMR

## 📋 Présentation

Ce projet implémente une chaîne de traitement distribué (ETL + ML) pour l'extraction de caractéristiques visuelles à partir d'images et leur réduction dimensionnelle via PCA (Principal Component Analysis).
Il exploite la puissance du calcul distribué sur AWS EMR avec des instances GPU pour traiter efficacement de larges volumes d'images.

### Cas d'usage

- **Analyse de collections d'images à grande échelle** : extraction de features pour classification, clustering ou recherche de similarité
- **Prétraitement ML** : génération de représentations vectorielles compactes pour des modèles downstream
- **Réduction de dimensionnalité** : compression de features haute dimension (1280D → K dimensions configurable)

---

## 🏗️ Architecture Technique

### Stack Technologique

- **Plateforme** : AWS EMR 7.12.0 (Amazon Linux 2023)
- **Orchestrateur** : Apache Spark (PySpark)
- **Deep Learning** : PyTorch 2.1.0 + CUDA 11.8
- **Modèle** : MobileNetV2 (ImageNet pré-entraîné, 1280 features)
- **Storage** : Amazon S3
- **Instances** : 
  - Master : m5.xlarge (CPU)
  - Workers : g4dn.xlarge (GPU NVIDIA T4)

### Pipeline de Traitement

```
Images S3 (JPG)
    ↓
Lecture binaire (Spark)
    ↓
Échantillonnage stratifié par catégorie
    ↓
Featurisation par lot (MobileNetV2 sur GPU)
    ↓
StandardScaler (normalisation)
    ↓
PCA (réduction dimensionnelle)
    ↓
Export Parquet (S3)
```

---

## 📁 Structure du Projet

```
.
├── bootstrap.sh                    # Script d'initialisation du cluster
├── cloud_deploy.py                 # Pipeline PySpark principal
├── lancement_cluster_et_job.sh     # Configuration et lancement EMR
└── README.md                       # Cette documentation
```

### Détail des Fichiers

#### 1. `bootstrap.sh`
Script exécuté au démarrage de chaque nœud du cluster pour installer :
- Outils système (`htop`)
- PyTorch 2.1.0 avec support CUDA 11.8
- Dépendances : Pillow, NumPy, Pandas, PyArrow

#### 2. `cloud_deploy.py`
Cœur du pipeline distribué.
- **Détection automatique de région S3**
- **Chargement optimisé** : Parquet-first avec fallback sur binaryFile
- **Échantillonnage stratifié** : N images par catégorie
- **Featurisation GPU** : Pandas UDF avec gestion d'alignement et d'erreurs
- **Normalisation** : StandardScaler avant PCA
- **PCA robuste** : Ajustement automatique de K selon les contraintes
- **Quality checks** : Validation NaN, vecteurs nuls, variance expliquée

#### 3. `lancement_cluster_et_job.sh`
Template de commande AWS CLI pour créer un cluster EMR avec :
- Bootstrap action
- Configuration réseau (VPC, Security Groups)
- Spot instances pour les workers
- Auto-termination après 1h d'inactivité
- Soumission automatique du job Spark

---

## ⚙️ Prérequis

### AWS

1. **Compte AWS** avec les services suivants activés :
   - EMR
   - EC2
   - S3
   - VPC

2. **IAM Roles** :
   - `AmazonEMR-ServiceRole-...` (service principal EMR)
   - `AmazonEMR-InstanceProfile-...` (profil EC2 pour les nœuds)

3. **Réseau** :
   - VPC avec subnet(s) configuré(s)
   - Security Groups pour Master et Slave
   - Clé EC2 (paire de clés SSH)

4. **S3 Bucket** avec structure :
   ```
   s3://mon-bucket/
   ├── Test/                    # Images d'entrée (JPG, organisées par dossiers)
   │   ├── categorie1/
   │   │   ├── image1.jpg
   │   │   └── ...
   │   └── categorie2/
   ├── bootstrap/
   │   └── bootstrap.sh
   ├── script/
   │   └── cloud_deploy.py
   └── logs/                    # Logs EMR (auto-créé)
   ```

### Local

- AWS CLI configuré (`aws configure`)
- Permissions IAM pour créer des clusters EMR

---

### Source des données à télécharger

https://www.kaggle.com/datasets/moltean/fruits

**1. Uploadez le dossier Test vers S3**
```bash
cd fruits-360/Test
aws s3 sync . s3://votre-bucket/Test/
```

**2. Uploadez les scripts du pipeline**
```bash
aws s3 cp bootstrap.sh s3://votre-bucket/bootstrap/
aws s3 cp cloud_deploy.py s3://votre-bucket/script/
```

---

### Configuration

Éditez `lancement_cluster_et_job.sh` et remplacez :

| Placeholder | Description | Exemple |
|-------------|-------------|---------|
| `<MON-BUCKET-S3>` | Nom de votre bucket S3 | `p9.data` |
| `<ID-COMPTE-AWS>` | ID de votre compte AWS | `123456789` |
| `<ID-SG-MASTER>` | Security Group Master | `sg-0abc123...` |
| `<ID-SG-SLAVE>` | Security Group Worker | `sg-0def456...` |
| `<MA-CLE-EC2>` | Nom de votre clé SSH | `my-keypair` |
| `<ID-SUBNET-VPC>` | Subnet ID | `subnet-0123...` |

### Variables d'Environnement (Optionnel)

Le pipeline supporte plusieurs variables d'environnement pour personnaliser l'exécution :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `P9_S3_ENDPOINT` | `s3.eu-west-3.amazonaws.com` | Endpoint S3 |
| `P9_S3_BUCKET` | `p9.data` | Bucket contenant les données |
| `P9_SAMPLES_PER_CATEGORY` | `45` | Nombre d'images par catégorie |
| `P9_PCA_K` | `128` | Dimensions PCA de sortie |
| `P9_TARGET_FILES` | `""` | Nombre de fichiers Parquet (vide = auto) |
| `P9_ARROW_BATCH` | `128` | Taille de batch Arrow pour Pandas UDF |
| `P9_LOG_LEVEL` | `INFO` | Niveau de log (DEBUG, INFO, WARNING) |

Pour les passer au job Spark, modifiez la section `--steps` :

```bash
"Args":["spark-submit",
        "--conf","spark.yarn.appMasterEnv.PYTHONUNBUFFERED=1",
        "--conf","spark.yarn.appMasterEnv.P9_PCA_K=256",
        "--conf","spark.executorEnv.P9_PCA_K=256",
        "s3://votre-bucket/script/cloud_deploy.py"]
```

### Lancement

```bash
bash lancement_cluster_et_job.sh
```

La commande retourne un `ClusterId`. Notez-le pour le monitoring.

### Monitoring

Surveillez l'exécution via :

1. **AWS Console** : EMR → Clusters → Votre cluster → Steps
2. **AWS CLI** :
   ```bash
   aws emr describe-cluster --cluster-id j-XXXXXXXXXXXXX
   aws emr list-steps --cluster-id j-XXXXXXXXXXXXX
   ```
3. **Logs S3** : `s3://votre-bucket/logs/`

### Récupération des résultats

Une fois le step complété (SUCCESS), les résultats sont dans :

```
s3://votre-bucket/fruit_categories_pca/
    part-00000-....parquet
    part-00001-....parquet
    ...
    _SUCCESS
```

Structure du Parquet :

| Colonne | Type | Description |
|---------|------|-------------|
| `path` | string | Chemin S3 de l'image source |
| `label` | string | Catégorie (nom du dossier) |
| `pca_0` | float | Composante PCA 0 |
| `pca_1` | float | Composante PCA 1 |
| ... | ... | ... |
| `pca_{K-1}` | float | Composante PCA K-1 |

---

## 📊 Logs et diagnostics

Le pipeline génère des logs structurés pour faciliter le debugging :

### Informations de démarrage
```
Spark version: 3.5.x
sc.defaultParallelism: 8
Partitions: 16
S3 Region (from endpoint): eu-west-3
EMR Cluster Region (detected): eu-west-3
```

### Probe GPU
```
--- GPU probe (executors) ---
host=ip-10-0-1-23 | GPU=T4 | cuda=True (cu=11.8)
host=ip-10-0-1-24 | GPU=T4 | cuda=True (cu=11.8)
```

### Statistiques d'images invalides
```
Invalid images (global): 3 / 450 (0.67%)
Top categories by invalid_rate (worst 5):
     label  total  invalid  invalid_rate
   Damaged     45        2      0.044444
```

### Variance expliquée
```
Total variance explained by 128 components: 0.9234
Component 0: 0.1245
Component 1: 0.0823
...
```

### Rapport Parquet
```
Files: 4 | Total: 12.34 MiB | Avg: 3.09 MiB | Min/Max: 2.98/3.15 MiB
```

---

## 🔧 Optimisations et bonnes pratiques

### Performance

1. **Parquet Compacté** : Le pipeline crée automatiquement une version optimisée des données lors de la première exécution (`Test_Compacted.parquet`). Les runs suivants seront 3-5x plus rapides.

2. **Partitionnement** : Le nombre de partitions est automatiquement ajusté selon `sc.defaultParallelism * 2`.

3. **Caching Stratégique** : Les DataFrames intermédiaires sont mis en cache pour éviter les recalculs.

4. **Batch Size Arrow** : Ajustez `P9_ARROW_BATCH` selon la RAM GPU disponible (128 = ~2GB VRAM).

### Coûts

1. **Spot Instances** : Le script utilise des instances Spot pour les workers (économie ~70%).

2. **Auto-termination** : Le cluster se termine après 1h d'inactivité (`IdleTimeout:3600`).

3. **Scale-down** : `TERMINATE_AT_TASK_COMPLETION` pour libérer les workers dès que possible.

### Robustesse

1. **Images Invalides** : Le pipeline skip automatiquement les images corrompues et log les statistiques.

2. **PCA Fallback** : Si `k` demandé est trop grand, le pipeline réduit automatiquement jusqu'à une valeur viable.

3. **Alignement Batch** : La featurisation préserve l'alignement avec les rows d'entrée, même en cas d'erreur partielle.

---

## Troubleshooting

### Problème : Bootstrap échoue
**Symptôme** : Le cluster passe en état TERMINATED_WITH_ERRORS  
**Solution** :
- Vérifiez que `bootstrap.sh` est accessible dans S3
- Consultez les logs : `s3://votre-bucket/logs/j-XXXXX/node/i-XXXXX/bootstrap-actions/`

### Problème : Out of Memory (OOM)
**Symptôme** : Executors crashent avec erreurs YARN  
**Solutions** :
- Réduisez `P9_ARROW_BATCH` (ex: 64 ou 32)
- Augmentez `spark.executor.memory` dans `--conf`
- Réduisez `P9_SAMPLES_PER_CATEGORY`

### Problème : PCA échoue
**Symptôme** : `PCA(k=...) failed`  
**Solutions** :
- Le pipeline ajuste automatiquement K
- Si échec persistant : nombre d'images < K demandé
- Augmentez `P9_SAMPLES_PER_CATEGORY` ou réduisez `P9_PCA_K`

### Problème : Permissions S3
**Symptôme** : `AccessDenied` dans les logs  
**Solution** :
- Vérifiez que l'InstanceProfile a les permissions S3 (GetObject, PutObject)
- Exemple de policy :
  ```json
  {
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
    "Resource": ["arn:aws:s3:::votre-bucket/*", "arn:aws:s3:::votre-bucket"]
  }
  ```


