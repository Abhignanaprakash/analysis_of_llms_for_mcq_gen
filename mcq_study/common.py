from __future__ import annotations
import json, platform, random, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch, yaml

INSTRUCTION = "Generate a multiple-choice question with four options and one correct answer from the following passage: {context}"

def load_config(path: str):
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")

def capture_environment():
    import transformers, peft
    data = {"captured_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
      "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
      "peft": peft.__version__, "cuda": torch.version.cuda, "gpu": None, "gpu_ram_gb": None}
    if torch.cuda.is_available():
        p=torch.cuda.get_device_properties(0); data.update(gpu=p.name, gpu_ram_gb=round(p.total_memory/2**30,2))
        try: data["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        except (OSError, subprocess.CalledProcessError): pass
    return data
