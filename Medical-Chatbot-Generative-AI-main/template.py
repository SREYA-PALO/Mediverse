import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_files = [
    "src/_init_.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "requirements.txt",
    "setup.py",
    "app.py",
    "research/trials.ipynb",
    "test.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir:  # Checks if filedir is not empty
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass  # Creates an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
