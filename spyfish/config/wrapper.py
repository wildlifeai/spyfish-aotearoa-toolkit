from spyfish.config.aws import AWSConfig
from spyfish.config.biigle import BiigleConfig
from spyfish.config.extraction import ExtractionConfig
from spyfish.config.ml import MLConfig
from spyfish.config.paths import PathsConfig
from spyfish.config.zooniverse import ZooniverseConfig


class ConfigWrapper(
    PathsConfig, MLConfig, BiigleConfig, ExtractionConfig, ZooniverseConfig, AWSConfig
):
    pass


config = ConfigWrapper()
