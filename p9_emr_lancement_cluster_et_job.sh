aws emr create-cluster \
 --name "p9-emr" \
 --log-uri "s3://<MON-BUCKET-S3>/logs" \
 --release-label "emr-7.12.0" \
 --service-role "arn:aws:iam::<ID-COMPTE-AWS>:role/service-role/AmazonEMR-ServiceRole-..." \
 --unhealthy-node-replacement \
 --ec2-attributes '{
    "InstanceProfile":"AmazonEMR-InstanceProfile-...",
    "EmrManagedMasterSecurityGroup":"<ID-SG-MASTER>",
    "EmrManagedSlaveSecurityGroup":"<ID-SG-SLAVE>",
    "KeyName":"<MA-CLE-EC2>",
    "SubnetIds":["<ID-SUBNET-VPC>"]
   }' \
 --tags 'for-use-with-amazon-emr-managed-policies=true' \
 --applications Name=Hadoop Name=Spark \
 --instance-groups '[
    {"InstanceCount":1,"InstanceGroupType":"MASTER","Name":"Primary","InstanceType":"m5.xlarge",...},
    {"BidPrice":"0.205","InstanceCount":2,"InstanceGroupType":"CORE","Name":"Core","InstanceType":"g4dn.xlarge",...}
   ]' \
 --bootstrap-actions '[{"Name":"bootstrap","Path":"s3://<MON-BUCKET-S3>/bootstrap/bootstrap.sh"}]' \
 --steps '[{"Name":"P9-Feature-Extraction",
            "ActionOnFailure":"CANCEL_AND_WAIT",
            "Jar":"command-runner.jar",
            "Args":["spark-submit",
                    "--conf","spark.yarn.appMasterEnv.PYTHONUNBUFFERED=1",
                    "--conf","spark.executorEnv.PYTHONUNBUFFERED=1",
                    "s3://<MON-BUCKET-S3>/script/cloud_deploy.py"]
          }]' \
 --scale-down-behavior "TERMINATE_AT_TASK_COMPLETION" \
 --ebs-root-volume-size "50" \
 --auto-termination-policy '{"IdleTimeout":3600}' \
 --region "eu-west-3"
