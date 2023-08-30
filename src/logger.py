import logging
import os
from datetime import datetime


LOG_FILE = f'{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.log'
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(os.path.dirname(logs_path), exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(filename=LOG_FILE_PATH,
                    level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s')



