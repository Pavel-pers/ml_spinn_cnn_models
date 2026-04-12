import json
import torch
from pathlib import Path
from hashlib import sha256
from typing import Dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperementsStorage:
    def __init__(self, exp_dir: Path):
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir = exp_dir

    def get_experement_id(self, exp_config: Dict):
        conf_string = json.dumps(exp_config, sort_keys=True).encode('utf-8')
        conf_hash = sha256(conf_string).hexdigest()[:8]
        return conf_hash

    def save_result(self, result: Dict):
        cur_cfg = result['config']
        cur_mse = result.get('mse')
        cur_l2_norm = result.get('l2_norm')
        cur_best_err = result.get('best_error')

        cur_id = self.get_experement_id(cur_cfg)
        cur_folder = self.exp_dir / cur_id
        if cur_folder.exists():
            print('[WARN] experemnt already exists')
        cur_folder.mkdir(parents=True, exist_ok=True)

        meta = {
            "config": cur_cfg,
            "mse": cur_mse,
            "l2_norm": cur_l2_norm,
            "best_error": cur_best_err,
        }
        with open(cur_folder / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        history_save = {iter_n: err for iter_n, err in result["history"].items()}
        torch.save(history_save, cur_folder / "history.pt")

        torch.save(result["model"].state_dict(), cur_folder / "model.pth")

        print(f'[INFO] saved in {cur_id}')

    def load_experement(self, config: Dict, model: torch.nn.Module):
        cur_id = self.get_experement_id(config)
        cur_folder = self.exp_dir / cur_id

        if not cur_folder.exists():
            return None

        with open(cur_folder / "meta.json", "r") as f:
            meta = json.load(f)

        cur_history = torch.load(cur_folder / "history.pt", weights_only=True)
        model.load_state_dict(torch.load(cur_folder / "model.pth", map_location=device))

        print(f'[INFO] loaded experement {cur_id}')

        return {
            "config": meta["config"],
            "mse": meta["mse"],
            "l2_norm": meta["l2_norm"],
            "best_err": meta["best_error"],
            "history": cur_history,
            "model": model,
        }


ARTIFACTS_DIR = Path("./artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EXPEREMENTS_DIR = ARTIFACTS_DIR / "experements"
EXPEREMENTS_DIR.mkdir(parents=True, exist_ok=True)
storage = ExperementsStorage(EXPEREMENTS_DIR)
