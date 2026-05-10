import logging
import sys


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        datefmt="%d-%m %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
