import logging
import sys
from pathlib import Path
from datetime import datetime
import os
import random
import numpy as np
import torch

class Constants:
    ETHIOPIC_RANGE_1 = (0x1200, 0x137F)
    ETHIOPIC_RANGE_2 = (0x1380, 0x139F)
    ETHIOPIC_RANGE_3 = (0x2D80, 0x2DDF)
    ETHIOPIC_RANGE_4 = (0xAB00, 0xAB2F)
    CUDA_FP16_MIN_COMPUTE = 7
    TROCR_DEFAULT_IMAGE_SIZE = 384
    LABEL_PAD_TOKEN = -100

def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Setup logging system."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"trocr_tigrinya_{timestamp}.log"
    
    logger = logging.getLogger("TrOCR_Tigrinya")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)