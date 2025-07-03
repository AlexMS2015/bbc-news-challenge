from pydantic import BaseModel
import yaml
from pathlib import Path
from loguru import logger
from typing import Any

class Config(BaseModel):
    random_state: int
    folders: dict[str, str]
    data: dict[str, Any]
    feature_eng: dict[str, dict]
    model: dict[str, Any]

    def make_path(self, folder: str) -> Path:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_path(self) -> Path:
        return self.make_path(self.folders['data'])
    
    @property
    def models_path(self) -> Path:
        return self.make_path(self.folders['models'])
    
    @property
    def eval_path(self) -> Path:
        return self.make_path(self.folders['eval'])


path = Path(__file__).parent.parent / "config.yaml"
with open(path, "r") as file:
    config_data = yaml.safe_load(file)
    
config = Config(**config_data)

logger.remove()
logger.add('app.log', level="DEBUG")