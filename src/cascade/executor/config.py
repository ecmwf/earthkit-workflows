# NOTE possibly make ext configurable
logging_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "format": "{asctime}:{levelname}:{name}:{process}:{message:1.10000}",
            "style": "{",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "uvicorn": {"level": "INFO"},
        "forecastbox": {"level": "DEBUG"},
        "forecastbox.worker": {"level": "DEBUG"},
        "forecastbox.executor": {"level": "DEBUG"},
        "cascade": {"level": "INFO"},
        "cascade.low": {"level": "DEBUG"},
        "cascade.shm": {"level": "DEBUG"},
        "cascade.controller": {"level": "DEBUG"},
        "cascade.executor": {"level": "DEBUG"},
        "cascade.scheduler": {"level": "DEBUG"},
        "cascade.gateway": {"level": "DEBUG"},
        "httpcore": {"level": "ERROR"},
        "httpx": {"level": "ERROR"},
        "": {"level": "WARNING", "handlers": ["default"]},
    },
}


def logging_config_filehandler(filename) -> dict:
    return {
        **logging_config,
        **{
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.FileHandler",
                    "filename": filename,
                },
            },
        },
    }
