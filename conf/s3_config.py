"""
Configuration for Yandex Cloud S3 bucket connection
"""
import os
from typing import Optional


class YandexS3Config:
    """Configuration for Yandex Cloud S3 access"""
    
    # Yandex Cloud S3 endpoint
    S3_ENDPOINT = "https://storage.yandexcloud.net"
    
    # Your credentials (set via environment variables for security)
    S3_ACCESS_KEY = os.getenv("YANDEX_S3_ACCESS_KEY", "")
    S3_SECRET_KEY = os.getenv("YANDEX_S3_SECRET_KEY", "")
    S3_BUCKET_NAME = os.getenv("YANDEX_S3_BUCKET_NAME", "")
    
    # Region (Yandex Cloud uses specific region)
    S3_REGION = "ru-central1"
    
    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """
        Initialize S3 configuration
        
        Args:
            access_key: AWS/Yandex S3 access key
            secret_key: AWS/Yandex S3 secret key
            bucket_name: S3 bucket name
            endpoint: S3 endpoint URL
            region: AWS region or Yandex Cloud region
        """
        self.access_key = access_key or self.S3_ACCESS_KEY
        self.secret_key = secret_key or self.S3_SECRET_KEY
        self.bucket_name = bucket_name or self.S3_BUCKET_NAME
        self.endpoint = endpoint or self.S3_ENDPOINT
        self.region = region or self.S3_REGION
        
        if not all([self.access_key, self.secret_key, self.bucket_name]):
            raise ValueError(
                "Missing required S3 credentials. Set environment variables: "
                "YANDEX_S3_ACCESS_KEY, YANDEX_S3_SECRET_KEY, YANDEX_S3_BUCKET_NAME"
            )
    
    def get_s3_url(self, prefix: str = "") -> str:
        """Get full S3 URL for a given prefix"""
        prefix = prefix.lstrip("/")
        return f"s3://{self.bucket_name}/{prefix}" if prefix else f"s3://{self.bucket_name}"
