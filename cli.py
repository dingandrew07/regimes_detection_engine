#!/usr/bin/env python3
"""
Command Line Interface for Regimes Replication Project

Usage examples:
    python cli.py factor-returns
    python cli.py state-variables
    python cli.py similarity-score --target-month 2022-01
    python cli.py backtest --start-date 1990-01-31 --vol-target 0.12
    python cli.py similar-periods --target-month 2008-10
    python cli.py appendix
"""

import sys
import os
import argparse
from pathlib import Path
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.state_variables.factor_returns import download_factor_data
from src.state_variables.state_variables import get_state_variables
from src.state_variables.similarity_score import calculate_similarity_scores
from src.backtest.back_test import run_backtest as execute_backtest
from src.similar_periods import find_similar_periods
from src.backtest.appendix import generate_appendix


def update_config(config, updates):
    """Update config with command line arguments"""
    for key, value in updates.items():
        if value is not None:
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value
    return config


def save_config_snapshot(config, output_path):
    """Save current config as a snapshot"""
    snapshot_path = Path(output_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(snapshot_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Config snapshot saved to: {snapshot_path}")


def run_factor_returns(args):
    """Run factor returns calculation"""
    print("🔄 Downloading factor data...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = update_config(config, vars(args))
    
    if args.save_config:
        save_config_snapshot(config, f"cache/config_factor_returns_{args.save_config}.yaml")
    
    download_factor_data(config)
    print("✅ Factor returns calculation completed!")


def run_state_variables(args):
    """Run state variables calculation"""
    print("🔄 Calculating state variables...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = update_config(config, vars(args))
    
    if args.save_config:
        save_config_snapshot(config, f"cache/config_state_vars_{args.save_config}.yaml")
    
    get_state_variables(config)
    print("✅ State variables calculation completed!")


def run_similarity_score(args):
    """Run similarity score calculation"""
    print("🔄 Calculating similarity scores...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = update_config(config, vars(args))
    
    if args.save_config:
        save_config_snapshot(config, f"cache/config_similarity_{args.save_config}.yaml")
    
    calculate_similarity_scores(config)
    print("✅ Similarity score calculation completed!")


def run_backtest(args):
    """Run backtest"""
    print("🔄 Running backtest...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = update_config(config, vars(args))
    
    if args.save_config:
        save_config_snapshot(config, f"cache/config_backtest_{args.save_config}.yaml")
    
    # Extract backtest parameters from config
    backtest_params = {
        'n_buckets': config['backtest'].get('n_buckets', 5),
        'back_test_start_date': config['backtest'].get('back_test_start_date', '1985-01-31'),
        'forward_look_months': config['backtest'].get('forward_look_months', 1),
        'similarity_window': config['state_variables']['similarity_score'].get('similarity_window', 1),
    }
    
    # Check if efficacy extension is enabled
    efficacy_config = config.get('extensions', {}).get('efficacy_score', {})
    use_efficacy = efficacy_config.get('enabled', False)
    
    if use_efficacy:
        print(f"📊 Efficacy extension enabled: {efficacy_config.get('bootstrap_iterations', 200)} bootstrap iterations")
    
    execute_backtest(**backtest_params, use_efficacy=use_efficacy, efficacy_config=efficacy_config)
    print("✅ Backtest completed!")


def run_similar_periods(args):
    """Run similar periods analysis"""
    print("🔄 Finding similar periods...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = update_config(config, vars(args))
    
    if args.save_config:
        save_config_snapshot(config, f"cache/config_similar_periods_{args.save_config}.yaml")
    
    find_similar_periods(config)
    print("✅ Similar periods analysis completed!")


def run_appendix(args):
    """Run appendix generation"""
    print("🔄 Generating appendix...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = update_config(config, vars(args))
    
    if args.save_config:
        save_config_snapshot(config, f"cache/config_appendix_{args.save_config}.yaml")
    
    generate_appendix(config)
    print("✅ Appendix generation completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Regimes Replication Project CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Factor Returns
    factor_parser = subparsers.add_parser('factor-returns', help='Download and process factor returns')
    factor_parser.add_argument('--save-config', help='Save config snapshot with this name')
    factor_parser.set_defaults(func=run_factor_returns)
    
    # State Variables
    state_parser = subparsers.add_parser('state-variables', help='Calculate state variables')
    state_parser.add_argument('--save-config', help='Save config snapshot with this name')
    state_parser.set_defaults(func=run_state_variables)
    
    # Similarity Score
    similarity_parser = subparsers.add_parser('similarity-score', help='Calculate similarity scores')
    similarity_parser.add_argument('--target-month', help='Target month (YYYY-MM)')
    similarity_parser.add_argument('--save-config', help='Save config snapshot with this name')
    similarity_parser.set_defaults(func=run_similarity_score)
    
    # Backtest
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--vol-target', type=float, help='Volatility target')
    backtest_parser.add_argument('--n-buckets', type=int, help='Number of buckets')
    backtest_parser.add_argument('--save-config', help='Save config snapshot with this name')
    backtest_parser.set_defaults(func=run_backtest)
    
    # Similar Periods
    periods_parser = subparsers.add_parser('similar-periods', help='Find similar periods')
    periods_parser.add_argument('--target-month', help='Target month (YYYY-MM)')
    periods_parser.add_argument('--save-config', help='Save config snapshot with this name')
    periods_parser.set_defaults(func=run_similar_periods)
    
    # Appendix
    appendix_parser = subparsers.add_parser('appendix', help='Generate appendix')
    appendix_parser.add_argument('--save-config', help='Save config snapshot with this name')
    appendix_parser.set_defaults(func=run_appendix)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 