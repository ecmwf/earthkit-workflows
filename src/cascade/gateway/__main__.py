import logging.config

import fire

from cascade.executor.config import logging_config
from cascade.gateway.server import serve


def main(url: str) -> None:
    logging.config.dictConfig(logging_config)
    serve(url)


if __name__ == "__main__":
    fire.Fire(main)
