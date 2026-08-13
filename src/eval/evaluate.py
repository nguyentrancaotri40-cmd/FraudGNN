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


def _truncate_array(arr, max_items=20):
    """
    Cắt ngắn array để hiển thị, giữ max_items phần tử đầu và cuối.
    """
    if not isinstance(arr, (list, tuple, np.ndarray)):
        return arr
    if len(arr) <= max_items * 2:
        return list(arr)
    
    arr_list = list(arr)
    first = arr_list[:max_items]
    last = arr_list[-max_items:]
    return first + ['...'] + last


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
    
    # ✅ CẮT NGẮN CÁC MẢNG DÀI để hiển thị hàng ngang
    for key in ['val_scores', 'test_scores', 'val_labels', 'test_labels']:
        if key in serializable_metrics and isinstance(serializable_metrics[key], list):
            original = serializable_metrics[key]
            if len(original) > 50:  # Nếu dài quá 50 phần tử
                # Lưu full data vào key riêng để không mất dữ liệu
                serializable_metrics[f'{key}_full'] = original
                # Hiển thị preview
                serializable_metrics[key] = _truncate_array(original, max_items=10)
    
    # ✅ Sử dụng numpy để in hàng ngang nếu là array
    # Định dạng numpy để in gọn
    np.set_printoptions(
        threshold=20,           # Chỉ in 20 phần tử
        edgeitems=5,            # In 5 phần tử đầu và 5 cuối
        linewidth=200,          # Dòng rộng để in hàng ngang
        suppress=True,          # Không dùng scientific notation
        precision=4             # Làm tròn 4 chữ số thập phân
    )
    
    import json as json_lib
    
    # Bước 1: Dump với indent=2
    temp_json = json_lib.dumps(serializable_metrics, ensure_ascii=False, indent=2, default=str)
    
    # Bước 2: Format lại các list thành 1 dòng
    import re
    
    def _format_array_line(match):
        """Format mảng số thành 1 dòng."""
        content = match.group(1)
        # Nếu là mảng số đơn giản
        if re.match(r'^[\d.\-, e]+$', content):
            # Làm sạch khoảng trắng
            cleaned = re.sub(r'\s+', ' ', content)
            return f'[{cleaned}]'
        return match.group(0)
    
    # Format tất cả các list
    lines = temp_json.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Nếu dòng bắt đầu bằng "["
        if line.strip().startswith('[') and line.strip().endswith(']'):
            result.append(line)
        elif line.strip().startswith('['):
            # List bị xuống dòng - gom lại
            full_line = line.strip()
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                full_line += ' ' + next_line
                i += 1
                if next_line.endswith(']'):
                    break
            result.append(full_line)
        else:
            result.append(line)
            i += 1
    
    # Đọc lại và format
    try:
        parsed = json_lib.loads('\n'.join(result))
        
        # Custom encoder để in list trên 1 dòng
        class CompactEncoder(json_lib.JSONEncoder):
            def encode(self, obj):
                if isinstance(obj, list):
                    # Nếu tất cả phần tử đều là số -> 1 dòng
                    if all(isinstance(x, (int, float)) for x in obj):
                        if len(obj) > 100:
                            # Cắt ngắn nếu quá dài
                            first = obj[:10]
                            last = obj[-10:]
                            return '[' + ', '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in first) + ', ..., ' + ', '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in last) + ']'
                        return '[' + ', '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in obj) + ']'
                    # Nếu list ngắn (< 30) -> 1 dòng
                    if len(obj) <= 30:
                        return '[' + ', '.join(self.encode(item) for item in obj) + ']'
                    # List dài -> xuống dòng
                    return super().encode(obj)
                elif isinstance(obj, dict):
                    items = []
                    for key, value in obj.items():
                        key_repr = json_lib.dumps(key, ensure_ascii=False)
                        value_repr = self.encode(value)
                        items.append(f'{key_repr}: {value_repr}')
                    return '{' + ', '.join(items) + '}'
                else:
                    if isinstance(obj, float):
                        return f'{obj:.4f}'
                    return json_lib.dumps(obj, ensure_ascii=False)
        
        # Ghi file
        with open(path, "w", encoding="utf-8") as f:
            json_lib.dump(
                parsed,
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
                cls=CompactEncoder
            )
        print(f"✅ Metrics saved with compact formatting to: {path}")
        
    except Exception as e:
        print(f"⚠️ Compact formatting failed: {e}. Using fallback.")
        with open(path, "w", encoding="utf-8") as f:
            json_lib.dump(
                serializable_metrics,
                f,
                ensure_ascii=False,
                indent=2,
                default=str
            )