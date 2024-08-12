import logging
import logging.config
import os

def configure_logging():
    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s %(levelname)s %(message)s',
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'filename': os.path.join(log_directory, 'app.log'),
                'formatter': 'default',
                'level': 'DEBUG',
            },
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': 'DEBUG',
            },
        },
        'root': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
        },
    })
