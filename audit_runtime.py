#!/usr/bin/env python3
"""Audit runtime values in JSON files."""

import json

files = [
    "outputs/results/hybrid_metrics_latest.json",
    "outputs/results/ieee_cis_baseline_30000_thr065_metrics.json",
    "outputs/results/ieee_cis_baseline_30000_thr070_metrics.json",
    "outputs/results/ieee_cis_baseline_50000_thr065_metrics.json",
    "outputs/results/ieee_cis_baseline_50000_thr070_metrics.json",
    "outputs/results/hybrid/ieee_cis_hybrid_unweighted_30000_thr065_metrics.json",
    "outputs/results/hybrid/ieee_cis_hybrid_unweighted_30000_thr070_metrics.json",
    "outputs/results/hybrid/ieee_cis_hybrid_unweighted_50000_thr065_metrics.json",
    "outputs/results/hybrid/ieee_cis_hybrid_unweighted_50000_thr070_metrics.json",
]

for filepath in files:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        runtime = data.get('runtime', {})

        data_loading = runtime.get('data_loading_sec', 0.0)
        data_splitting = runtime.get('data_splitting_sec', 0.0)
        preprocessing = runtime.get('preprocessing_sec', 0.0)
        graph_building = runtime.get('graph_building_sec', 0.0)
        tssgc_training = runtime.get('tssgc_training_sec', 0.0)
        inference = runtime.get('inference_sec', 0.0)
        rl_training = runtime.get('rl_training_sec', 0.0)

        expected_sum = data_loading + data_splitting + preprocessing + graph_building + tssgc_training + inference + rl_training
        actual_total = runtime.get('total_runtime_sec', 0.0)
        diff = expected_sum - actual_total
        ratio = (actual_total / expected_sum * 100) if expected_sum > 0 else 0

        print(f"\n{filepath}:")
        print(f"  data_loading: {data_loading}")
        print(f"  data_splitting: {data_splitting}")
        print(f"  preprocessing: {preprocessing}")
        print(f"  graph_building: {graph_building}")
        print(f"  tssgc_training: {tssgc_training}")
        print(f"  inference: {inference}")
        print(f"  rl_training: {rl_training}")
        print(f"  Expected sum: {expected_sum:.6f}")
        print(f"  Actual total: {actual_total:.6f}")
        print(f"  Diff: {diff:.6f} ({ratio:.2f}%)")

        # Check runtime_per_sample
        runtime_per_sample = runtime.get('runtime_per_sample_sec', 0.0)
        expected_runtime_per_sample = expected_sum / 50000 if expected_sum > 0 else 0.0
        expected_runtime_per_sample_diff = runtime_per_sample - expected_runtime_per_sample
        print(f"  runtime_per_sample (expected): {expected_runtime_per_sample:.6f}")
        print(f"  runtime_per_sample (actual): {runtime_per_sample:.6f}")
        print(f"  Diff: {expected_runtime_per_sample_diff:.6f}")

        # Check throughput
        throughput = runtime.get('throughput_samples_per_sec', 0.0)
        expected_throughput = 50000 / expected_sum if expected_sum > 0 else 0.0
        expected_throughput_diff = throughput - expected_throughput
        print(f"  throughput (expected): {expected_throughput:.6f}")
        print(f"  throughput (actual): {throughput:.6f}")
        print(f"  Diff: {expected_throughput_diff:.6f}")

    except Exception as e:
        print(f"\n{filepath}: ERROR - {e}")
