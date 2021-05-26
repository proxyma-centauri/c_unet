import logging
import logging.config
import yaml

def configure_and_return_logger(filename:str):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)
    return logger
