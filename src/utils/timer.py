# src/utils/timer.py
from __future__ import annotations

import time
import functools
from typing import Any, Callable, Dict, Optional
import psutil
import torch
import warnings


def timer(func: Callable) -> Callable:
    """Decorator để đo thời gian chạy của hàm."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        
        if isinstance(result, dict):
            result['_timing'] = result.get('_timing', {})
            result['_timing'][func.__name__] = elapsed
        
        return result
    return wrapper


def measure_latency(model, data, device: str = "cpu", num_runs: int = 100) -> Dict[str, float]:
    """Đo latency (thời gian inference) của model.
    
    Args:
        model: PyTorch model
        data: Dữ liệu đầu vào (batch)
        device: 'cpu' hoặc 'cuda'
        num_runs: Số lần chạy để tính trung bình
    
    Returns:
        Dict với các metrics:
        - latency_mean_ms: Trung bình (ms)
        - latency_std_ms: Độ lệch chuẩn (ms)
        - latency_min_ms: Nhỏ nhất (ms)
        - latency_max_ms: Lớn nhất (ms)
        - latency_p50_ms: Percentile 50 (ms)
        - latency_p95_ms: Percentile 95 (ms)
        - latency_p99_ms: Percentile 99 (ms)
        - throughput_per_sec: Throughput (samples/s)
    """
    model.eval()
    data = data.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(data)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Đo latency
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(data)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    latencies = sorted(latencies)
    num_samples = data.x.size(0) if hasattr(data, 'x') else 1
    
    return {
        "latency_mean_ms": float(sum(latencies) / len(latencies)),
        "latency_std_ms": float((sum((x - sum(latencies)/len(latencies))**2 for x in latencies) / len(latencies)) ** 0.5),
        "latency_min_ms": float(latencies[0]),
        "latency_max_ms": float(latencies[-1]),
        "latency_p50_ms": float(latencies[int(len(latencies) * 0.5)]),
        "latency_p95_ms": float(latencies[int(len(latencies) * 0.95)]),
        "latency_p99_ms": float(latencies[int(len(latencies) * 0.99)]),
        "throughput_per_sec": float(1000 / (sum(latencies) / len(latencies)) * num_samples),
    }


def get_memory_usage(alert_threshold: float = 0.85) -> Dict[str, float]:
    """Lấy thông tin memory usage và cảnh báo nếu gần OOM."""
    mem = psutil.virtual_memory()
    
    result = {
        "ram_total_gb": mem.total / (1024**3),
        "ram_available_gb": mem.available / (1024**3),
        "ram_used_gb": mem.used / (1024**3),
        "ram_usage_percent": mem.percent,
    }
    
    # ✅ Cảnh báo nếu memory quá cao
    if mem.percent > alert_threshold * 100:
        warnings.warn(
            f"⚠️ High memory usage: {mem.percent:.1f}% "
            f"(threshold: {alert_threshold*100:.0f}%)",
            ResourceWarning
        )
    
    if torch.cuda.is_available():
        result["vram_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        result["vram_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        result["vram_max_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        
        # ✅ Cảnh báo nếu VRAM cao
        vram_percent = result["vram_allocated_gb"] / result.get("vram_total_gb", 1)
        if vram_percent > 0.85:
            warnings.warn(
                f"⚠️ High GPU memory: {result['vram_allocated_gb']:.2f}GB",
                ResourceWarning
            )
    
    return result


def format_time(seconds: float) -> str:
    """Format seconds thành string dễ đọc."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


def print_timing_summary(timing: Dict[str, float]):
    """In bảng timing summary đẹp."""
    print("\n" + "="*60)
    print("⏱️ TIMING SUMMARY")
    print("="*60)
    
    # Định nghĩa thứ tự và tên hiển thị
    order = [
        ("data_loading_sec", "Data Loading"),
        ("data_splitting_sec", "Data Splitting"),
        ("preprocessing_sec", "Preprocessing"),
        ("graph_building_sec", "Graph Building"),
        ("federated_training_sec", "Federated Training"),
        ("tssgc_training_sec", "TSSGC Training"),
        ("rl_training_sec", "RL Training"),
        ("inference_sec", "Inference"),
        ("total_runtime_sec", "TOTAL"),
    ]
    
    for key, name in order:
        if key in timing:
            value = timing[key]
            if key == "total_runtime_sec":
                print(f"{name:20} {format_time(value):>15}")
            else:
                print(f"{name:20} {format_time(value):>15}")
    
    # Throughput
    if "throughput_samples_per_sec" in timing:
        print(f"{'Throughput':20} {timing['throughput_samples_per_sec']:>15.2f} samples/s")
    
    # Runtime per sample
    if "runtime_per_sample_sec" in timing:
        print(f"{'Per sample':20} {timing['runtime_per_sample_sec']*1000:>15.2f} ms")
    
    print("="*60)