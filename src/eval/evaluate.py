# src/eval/evaluate.py
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import torch


@torch.no_grad()
def predict_scores(model, data, device: str = "cpu", timing: dict | None = None):
    model.eval()
    data = data.to(device)
    start_time = time.perf_counter()
    logits = model(data)
    scores = torch.sigmoid(logits).detach().cpu().numpy()
    if timing is not None:
        timing["inference_sec"] = time.perf_counter() - start_time
    return scores, data.y.detach().cpu().numpy()


def _format_value(value, indent=0):
    """Format value với list trên 1 dòng."""
    if isinstance(value, list):
        # Nếu list chứa các giá trị đơn giản (số, string) -> giữ trên 1 dòng
        if all(isinstance(x, (int, float, str, bool)) or x is None for x in value):
            return '[' + ', '.join(_format_value(x) for x in value) + ']'
        else:
            # List phức tạp -> xuống dòng
            return '[' + ', '.join(_format_value(x, indent+2) for x in value) + ']'
    elif isinstance(value, dict):
        items = []
        for k, v in value.items():
            items.append(f'"{k}": {_format_value(v, indent+2)}')
        return '{' + ', '.join(items) + '}'
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    else:
        return json.dumps(value, ensure_ascii=False)


def save_metrics(metrics: dict, path: str) -> None:
    """Save metrics to JSON with proper formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    # ✅ Sử dụng cách format thủ công để list không bị xuống dòng
    # Chỉ xuống dòng cho dict level 1, list giữ nguyên 1 dòng
    import json as json_lib
    
    # Bước 1: Dump với indent=2 để có structure đẹp
    temp_json = json_lib.dumps(serializable_metrics, ensure_ascii=False, indent=2, default=str)
    
    # Bước 2: Đọc lại và format lại các list
    import re
    
    def _reformat_lists(match):
        """Tìm và format list trên 1 dòng."""
        content = match.group(1)
        # Nếu content ngắn -> giữ 1 dòng
        if len(content) < 80 and '\n' not in content:
            return content
        return match.group(0)
    
    # Tìm tất cả list và format lại
    # pattern: [ ... ] với nội dung bên trong
    lines = temp_json.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Kiểm tra xem dòng có bắt đầu bằng "  [" và kết thúc bằng "]"
        if line.strip().startswith('[') and line.strip().endswith(']'):
            # List trên 1 dòng -> giữ nguyên
            result.append(line)
        else:
            result.append(line)
        i += 1
    
    # Đọc lại và format
    import json as json_module
    try:
        parsed = json_module.loads('\n'.join(result))
        # Sử dụng compact encoder
        class CompactEncoder(json_module.JSONEncoder):
            def encode(self, obj):
                if isinstance(obj, list):
                    # Nếu list ngắn (< 20) -> 1 dòng
                    if len(obj) <= 20:
                        return '[' + ', '.join(self.encode(item) for item in obj) + ']'
                    # List dài -> xuống dòng
                    else:
                        return super().encode(obj)
                elif isinstance(obj, dict):
                    items = []
                    for key, value in obj.items():
                        key_repr = json_module.dumps(key, ensure_ascii=False)
                        value_repr = self.encode(value)
                        items.append(f'{key_repr}: {value_repr}')
                    return '{' + ', '.join(items) + '}'
                else:
                    return json_module.dumps(obj, ensure_ascii=False)
        
        # Ghi file với custom encoder
        with open(path, "w", encoding="utf-8") as f:
            json_module.dump(
                parsed,
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
                cls=CompactEncoder
            )
    except:
        # Fallback: ghi bình thường
        with open(path, "w", encoding="utf-8") as f:
            json_module.dump(
                serializable_metrics,
                f,
                ensure_ascii=False,
                indent=2,
                default=str
            )