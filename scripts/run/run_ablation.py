#!/usr/bin/env python3
"""
Chạy ablation study cho FraudGNN-RL vs FraudGNN-RL+
"""

import argparse
import json
import subprocess
import tempfile
import yaml
from pathlib import Path
import sys
import os

ABLATION_CONFIGS = {
    'baseline': {
        'hard_edges': True,
        'soft_edges': False,
        'hybrid_graph': False,
        'weighted_fusion': False,
        'federated': True,
        'dqn': True,
        'pruning': False,
    },
    'soft_only': {
        'hard_edges': False,
        'soft_edges': True,
        'hybrid_graph': False,
        'weighted_fusion': False,
        'federated': True,
        'dqn': True,
        'pruning': False,
    },
    'hybrid_unweighted': {
        'hard_edges': True,
        'soft_edges': True,
        'hybrid_graph': True,
        'weighted_fusion': False,
        'federated': True,
        'dqn': True,
        'pruning': False,
    },
    'hybrid_weighted': {
        'hard_edges': True,
        'soft_edges': True,
        'hybrid_graph': True,
        'weighted_fusion': True,
        'federated': True,
        'dqn': True,
        'pruning': False,
    },
    'full': {
        'hard_edges': True,
        'soft_edges': True,
        'hybrid_graph': True,
        'weighted_fusion': True,
        'federated': True,
        'dqn': True,
        'pruning': True,
    },
    'pruning_only': {
        'hard_edges': True,
        'soft_edges': False,
        'hybrid_graph': False,
        'weighted_fusion': False,
        'federated': True,
        'dqn': True,
        'pruning': True,
    },
}

ABLATION_QUESTIONS = {
    'baseline': 'Hard edges co du tot khong?',
    'soft_only': 'Causal structure co can thiet khong?',
    'hybrid_unweighted': 'Them soft (ngang hang) co giup?',
    'hybrid_weighted': 'Trong so hoa co giup them?',
    'full': 'Weighted + Prune — hieu ung cong don',
    'pruning_only': 'Pruning tu no co tac dung?',
}


def run_with_flags(config_path: str, ablation_name: str, flags: dict) -> dict:
    """Chay 1 ablation variant voi flags cu the."""
    
    print(f"   [DEBUG] Reading config: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['flags'] = flags
    config['experiment']['name'] = f"{config['experiment']['name']}_{ablation_name}"
    config['experiment']['ablation'] = ablation_name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name
        print(f"   [DEBUG] Temp config: {tmp_path}")
    
    # Set environment to include PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())
    
    cmd = [sys.executable, '-m', 'src.main_pipeline', '--config', tmp_path]
    print(f"   [DEBUG] CMD: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink()
        print(f"   ❌ Timeout after 600s")
        return {'error': 'Timeout', 'stdout': '', 'stderr': ''}
    
    # Xoa file tam
    try:
        Path(tmp_path).unlink()
    except:
        pass
    
    # In chi tiet loi
    if result.returncode != 0:
        print(f"   ❌ Return code: {result.returncode}")
        if result.stderr:
            print(f"   ❌ STDERR:\n{result.stderr[:1000]}")
        if result.stdout:
            print(f"   📝 STDOUT:\n{result.stdout[:500]}")
        return {
            'error': result.stderr or 'Unknown error',
            'stdout': result.stdout,
            'returncode': result.returncode
        }
    
    # Parse JSON tu output
    try:
        output = result.stdout
        # Tim JSON trong output
        start = output.find('{')
        end = output.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = output[start:end]
            return json.loads(json_str)
        else:
            print(f"   ⚠️ No JSON found in output")
            print(f"   Output preview: {output[:200]}")
            return {'error': 'No JSON found', 'stdout': output}
    except json.JSONDecodeError as e:
        print(f"   ⚠️ JSON parse error: {e}")
        return {'error': f'JSON parse error: {e}', 'stdout': result.stdout}
    except Exception as e:
        print(f"   ⚠️ Unexpected error: {e}")
        return {'error': str(e), 'stdout': result.stdout}


def print_summary_table(results: dict):
    """In bang so sanh cac ablation."""
    
    print("\n" + "="*100)
    print("ABLATION STUDY SUMMARY")
    print("="*100)
    
    print(f"{'Config':<22} {'AUC-ROC':<10} {'AUC-PR':<10} {'F1':<10} {'Recall':<10} {'FPR':<10} {'Question'}")
    print("-"*100)
    
    for name, result in results.items():
        if result and isinstance(result, dict) and 'test_metrics' in result:
            m = result['test_metrics']
            question = ABLATION_QUESTIONS.get(name, '')
            print(f"{name:<22} {m.get('auc_roc', 0):<10.4f} {m.get('auc_pr', 0):<10.4f} "
                  f"{m.get('f1', 0):<10.4f} {m.get('recall', 0):<10.4f} {m.get('fpr', 0):<10.4f} {question}")
        else:
            err_msg = result.get('error', 'ERROR') if isinstance(result, dict) else 'ERROR'
            print(f"{name:<22} {err_msg[:10]:<10} {err_msg[:10]:<10} {err_msg[:10]:<10} {err_msg[:10]:<10} {err_msg[:10]:<10}")
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Base config file')
    parser.add_argument('--ablation', type=str, choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument('--all', action='store_true', help='Run all ablations')
    parser.add_argument('--output', type=str, default='outputs/results/ablation_summary.json')
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    args = parser.parse_args()
    
    results = {}
    
    if args.all:
        for name, flags in ABLATION_CONFIGS.items():
            print(f"\n▶ Running ablation: {name}")
            print(f"   Question: {ABLATION_QUESTIONS.get(name, '')}")
            print(f"   Flags: {flags}")
            results[name] = run_with_flags(args.config, name, flags)
            print(f"   Result: {results[name].get('test_metrics', {}).get('auc_roc', 'ERROR') if isinstance(results[name], dict) else 'ERROR'}")
        
        # Luu ket qua
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {args.output}")
        
        print_summary_table(results)
        
    else:
        if args.ablation:
            flags = ABLATION_CONFIGS[args.ablation]
            result = run_with_flags(args.config, args.ablation, flags)
            print(json.dumps(result, indent=2))
        else:
            parser.print_help()


if __name__ == '__main__':
    main()