from pydantic import BaseModel
import yaml
from pathlib import Path
from loguru import logger

class Config(BaseModel):
    data: dict[str, str]

path = Path(__file__).parent.parent / "config.yaml"
with open(path, "r") as file:
    config_data = yaml.safe_load(file)
    
config = Config(**config_data)

logger.remove()
logger.add('app.log', level="INFO")