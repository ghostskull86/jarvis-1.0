import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import datetime

def setup_logging(log_dir: Path):
    """Menyiapkan konfigurasi logging dasar"""
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "jarvis.log",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_input(input_data: Any, expected_type: type) -> bool:
    """Validasi tipe data input"""
    return isinstance(input_data, expected_type)

def save_to_file(data: Dict, file_path: Path, indent: int = 4):
    """
    Menyimpan dictionary ke file JSON
    :param data: Data yang akan disimpan
    :param file_path: Path tujuan file
    :param indent: Indentasi untuk format JSON
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
        return True
    except Exception as e:
        logging.error(f"Error saving file {file_path}: {e}", exc_info=True)
        return False

def load_from_file(file_path: Path) -> Optional[Dict]:
    """
    Memuat data dari file JSON
    :return: Dictionary atau None jika gagal
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}", exc_info=True)
        return None

def current_timestamp() -> str:
    """Mendapatkan timestamp saat ini dalam format string"""
    return datetime.datetime.now().isoformat()