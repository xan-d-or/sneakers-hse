import boto3
import os

# ============================================================================
# 4. S3 КЛИЕНТ (self-contained)
# ============================================================================

class YandexS3Client:
    """Клиент для работы с Yandex Cloud S3"""

    def __init__(self, access_key, secret_key, bucket_name, endpoint, region):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            # endpoint_url=endpoint,
            # region_name=region,
            # aws_access_key_id=access_key,
            # aws_secret_access_key=secret_key,
        )
    
    def list_objects(self, prefix=""):
        """Список объектов в S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
            )
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception as e:
            print(f"✗ Ошибка при получении списка: {e}")
            return []
    
    def get_bucket_size(self, prefix=""):
        """Получить размер данных"""
        total_size = 0
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            for page in pages:
                total_size += sum(obj["Size"] for obj in page.get("Contents", []))
            return total_size
        except Exception:
            return 0

