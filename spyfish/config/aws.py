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

aws_config = AWSConfig()
