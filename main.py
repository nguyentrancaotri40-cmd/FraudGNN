# main.py — entry point gốc:  python main.py --config configs/paysim.yaml [--cv] [--seed 42]
import sys

for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

print(f"[GPU] CUDA available: {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

from src.main_pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
