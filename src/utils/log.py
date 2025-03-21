import json
import logging
import pickle
from pathlib import Path
import lightning.pytorch as pl


class JsonLogger:
    def __init__(self, pl_class: pl.LightningModule):
        try:
            save_dir = pl_class.logger.log_dir
        except:
            save_dir = None

        if save_dir != None:
            self.log_path = Path(save_dir) / 'outputs.json'
        else:
            self.log_path = Path('temp_debug_log.json')
    
    def log(self, message: dict):
        json_message = json.dumps(message, indent=2)
        with self.log_path.open('a') as f:
            f.write(json_message + '\n')


class PickleLogger:
    def __init__(self, pl_class: pl.LightningModule, log_dir=None):
        if log_dir:
            self.log_dir = Path(log_dir).expanduser()
        else:
            try:
                exp_dir = pl_class.logger.log_dir
            except:
                self.log_dir = Path('outputs')
            else:
                self.log_dir = Path(exp_dir) / 'outputs'
        
        self.log_dir.mkdir(exist_ok=True)
    
    def log(self, data: dict, file_name: str):
        save_path = self.log_dir / file_name
        with save_path.open('wb') as f:
            pickle.dump(data, f)


def setup_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(console_handler)
    if log_file:
        logger.addHandler(file_handler)

    return logger
