import os

from spyfish.config.base import BaseConfig


class AWSConfig(BaseConfig):
    @property
    def access_key_id(self):
        return os.getenv("AWS_ACCESS_KEY_ID")

    @property
    def secret_access_key(self):
        return os.getenv("AWS_SECRET_ACCESS_KEY")

    @property
    def region(self):
        return os.getenv("AWS_REGION", "eu-central-1")

    @property
    def s3_bucket(self) -> str:
        bucket = os.getenv("S3_BUCKET")
        if not bucket:
            raise ValueError("S3_BUCKET environment variable is not set.")
        return bucket


aws_config = AWSConfig()
