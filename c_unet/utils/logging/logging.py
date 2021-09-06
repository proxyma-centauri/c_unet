import logging
import logging.config
import yaml

def configure_and_return_logger(filename:str):
    """
    Loads the logging configuration and creates a logger instance

    Args:
        - filename (str): dilation that would haveen used by a transposed convolution
    Returns:
        Logger object for the submodule
    """
    with open(filename, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)
    return logger
