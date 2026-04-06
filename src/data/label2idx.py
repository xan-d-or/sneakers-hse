import boto3
from typing import Dict


def build_label2idx_s3(bucket: str, prefix: str) -> Dict[str, int]:
    """
    bucket: "my-bucket"
    prefix: "dataset/train/"
    """

    s3 = boto3.client("s3")

    class_names = set()

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]  # например: dataset/train/cat/img1.jpg

            parts = key.split("/")
            if len(parts) >= 2:
                class_name = parts[-2]  # папка перед файлом
                class_names.add(class_name)

    class_names = sorted(class_names)
    return {cls: i for i, cls in enumerate(class_names)}