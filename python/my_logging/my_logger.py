
import logging


def setup_default_logger():
    logging.basicConfig(
        filename="app.log",
        encoding="utf-8",
        filemode="w",
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.DEBUG
    )

    return logging

