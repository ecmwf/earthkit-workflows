base = {
    "version": 1,
    "disable_existing_loggers": True,
    "loggers": {
        "uvicorn": {"level": "INFO"},
        "forecastbox": {"level": "DEBUG"},
        "forecastbox.worker": {"level": "DEBUG"},
        "forecastbox.executor": {"level": "DEBUG"},
        "cascade": {"level": "INFO"},
        "cascade.main": {"level": "DEBUG"},
        "cascade.low": {"level": "DEBUG"},
        "cascade.shm": {"level": "DEBUG"},
        "cascade.controller": {"level": "DEBUG"},
        "cascade.executor": {"level": "DEBUG"},
        "cascade.scheduler": {"level": "DEBUG"},
        "cascade.gateway": {"level": "DEBUG"},
        "earthkit.workflows": {"level": "DEBUG"},
        "httpcore": {"level": "ERROR"},
        "httpx": {"level": "ERROR"},
        "": {"level": "WARNING", "handlers": ["default"]},
    },
}

handlers = {
    "stdout": {
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
    },
    "filename": lambda filename: {
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.FileHandler",
                    "filename": filename,
                },
            },
        },
}

formatters = {
    "line": {
        "formatters": {
            "default": {
                "format": "{asctime}:{levelname}:{name}:{process}:{message:1.10000}",
                "style": "{",
            },
        },
    },
    "json": {
        "formatters": {
            "default": {
               "()": "__main__.JSONFormatter", 
            },
        },
    },
}
