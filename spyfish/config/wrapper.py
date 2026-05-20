from spyfish.config.aws import AWSConfig
from spyfish.config.biigle import BiigleConfig
from spyfish.config.columns import ColumnsConfig
from spyfish.config.extraction import ExtractionConfig
from spyfish.config.ml import MLConfig
from spyfish.config.paths import PathsConfig
from spyfish.config.validation import ValidationConfig
from spyfish.config.zooniverse import ZooniverseConfig


class ConfigWrapper(
    PathsConfig,
    ColumnsConfig,
    ValidationConfig,
    MLConfig,
    BiigleConfig,
    ExtractionConfig,
    ZooniverseConfig,
    AWSConfig,
):
    pass


config = ConfigWrapper()
