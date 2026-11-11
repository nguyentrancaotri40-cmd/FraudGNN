# audit_runtime.py — kiểm tra môi trường + dữ liệu trước khi chạy:  python audit_runtime.py --config configs/paysim.yaml
import argparse
import platform
import sys
from pathlib import Path

import psutil
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args()
    print(f"Python {platform.python_version()} | {platform.platform()}")
    print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()}"
          + (f" | {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GB" if torch.cuda.is_available() else ""))
    try:
        import torch_geometric
        print(f"torch_geometric {torch_geometric.__version__}")
    except Exception as e:  # noqa
        print(f"[FAIL] torch_geometric: {e!r}")
    vm = psutil.virtual_memory()
    print(f"RAM {vm.total/2**30:.1f}GB (còn {vm.available/2**30:.1f}GB)")
    if a.config:
        from src.utils.config import load_config
        cfg = load_config(a.config)
        ds = cfg["dataset"]
        paths = [ds[k] for k in ("path", "transaction_path", "identity_path") if ds.get(k)]
        for p in paths:
            f = Path(p)
            print(f"[{'OK' if f.exists() else 'MISSING'}] {p}" + (f" ({f.stat().st_size/2**20:.0f}MB)" if f.exists() else ""))
        g = cfg["graph"]
        print(f"graph cap={g.get('max_neighbors_per_node')} seq_cap={cfg['model'].get('temporal_max_seq_len')} "
              f"batch_mode={cfg['train']['batch_mode']} flags={cfg['flags']}")
        if g.get("max_neighbors_per_node") is None and ds.get("sample_frac") in (None, 1.0):
            print("[WARN] Không cap láng giềng + toàn bộ dữ liệu: số cạnh có thể nổ bộ nhớ (xem README).")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
