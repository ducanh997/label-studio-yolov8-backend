import logging.config
import os

logging.config.dictConfig({
    "version": 1,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
            "formatter": "standard"
        }
    },
    "root": {
        "level": "ERROR",
        "handlers": [
            "console"
        ],
        "propagate": True
    }
})

from label_studio_ml.api import init_app
from model import YOLOv8Model

app = init_app(
    model_class=YOLOv8Model,
    model_dir=os.environ.get('MODEL_DIR', os.path.dirname(__file__)),
    redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
    redis_host=os.environ.get('REDIS_HOST', 'localhost'),
    redis_port=os.environ.get('REDIS_PORT', 6379)
)
