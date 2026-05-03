import os
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

class S3Client:
    def __init__(self,
                 service_name='s3',
                 endpoint_url="https://storage.yandexcloud.net",
                 aws_access_key_id='',
                 aws_secret_access_key=''
                 ) -> None:
        self.client = boto3.client(
        service_name=service_name,
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    def _download_one(self, bucket_name, s3_key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.client.download_file(bucket_name, s3_key, local_path)
        return local_path


    def download_folder_from_s3_parallel(
        self,
        bucket_name: str,
        s3_prefix: str,
        local_folder: str,
        max_workers: int = 16
    ):
        """
        Параллельная загрузка папки из S3
        """
        paginator = self.client.get_paginator("list_objects_v2")

        tasks = []

        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                s3_key = obj["Key"]

                if s3_key.endswith("/"):
                    continue

                relative_path = os.path.relpath(s3_key, s3_prefix)
                local_path = os.path.join(local_folder, relative_path)

                tasks.append((s3_key, local_path))

        print(f"Total files: {len(tasks)}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._download_one, bucket_name, s3_key, local_path)
                for s3_key, local_path in tasks
            ]

            for future in as_completed(futures):
                try:
                    path = future.result()
                    print(f"Downloaded: {path}")
                except Exception as e:
                    print(f"Error: {e}")

    def _upload_one(self, local_path, bucket_name, s3_key):
        self.client.upload_file(local_path, bucket_name, s3_key)
        return s3_key


    def upload_folder_to_s3_parallel(
        self,
        local_folder: str,
        bucket_name: str,
        s3_prefix: str = "",
        max_workers: int = 16
    ):
        """
        Параллельная загрузка папки в S3
        """
        tasks = []

        for root, _, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)

                relative_path = os.path.relpath(local_path, local_folder)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

                tasks.append((local_path, s3_key))

        print(f"Total files: {len(tasks)}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._upload_one, local_path, bucket_name, s3_key)
                for local_path, s3_key in tasks
            ]

            for future in as_completed(futures):
                try:
                    s3_key = future.result()
                    print(f"Uploaded: {s3_key}")
                except Exception as e:
                    print(f"Error: {e}")