import os
from pathlib import Path
from data_utils.data_reader import Dataset
import json
import random
random.seed(42)


SPLITS_PCTS = {
    "train": 0.7,
    "dev": 0.15,
    "test": 0.15
}

class GPTNegochat(Dataset):
    def __init__(self, data_path="datasets/GPT-NegoChat-Corpus/data", split='train'):
        super().__init__()
        if split not in SPLITS_PCTS:
            raise ValueError("Split {} not supported for GPTNegochat".format(split))
        files = list(Path(data_path).glob("*.json"))
        random.shuffle(files)
        train_files = files[:int(len(files) * SPLITS_PCTS["train"])]
        dev_files = files[int(len(files) * SPLITS_PCTS["train"]):int(len(files) * (SPLITS_PCTS["train"] + SPLITS_PCTS["dev"]))]
        test_files = files[int(len(files) * (SPLITS_PCTS["train"] + SPLITS_PCTS["dev"])):]
        files = {
            "train": train_files,
            "dev": dev_files,
            "test": test_files
        }
        
        for file in files[split]:
            dial = json.load(open(file, "r"))
            self.examples.append({
                "dialogue_id": file.stem,
                "turns": dial["turns"]
            })
