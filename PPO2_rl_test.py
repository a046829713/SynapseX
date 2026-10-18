import os
import sys
import argparse
from Brain.PPO2.lib.engine import PPO2Engine


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PPO2 Dedicated Testing & Order Export Runner"
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        default=False,
        help="Run in Formal Live Mode (generates formal_if_order_map.json for TradingSysteam.py)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing checkpoints (e.g., saves/ or Brain/PPO2/Meta)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Custom list of checkpoint .pt file paths",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Custom list of symbols to evaluate (e.g. BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ppo2_orders",
        help="Directory to export order CSVs and manifest files",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/ppo_v2.yaml",
        help="Path to training/testing YAML configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 70)
    print(" PPO2 Dedicated Testing & Order Export Environment")
    print(f" Mode: {'Formal Live Execution' if args.formal else 'Historical Backtest Orders Export'}")
    print(f" Config: {args.config}")
    print(f" Output Directory: {args.output_dir}")
    print("=" * 70)

    # 1. Initialize PPO2Engine
    engine = PPO2Engine(formal=args.formal, config=args.config)

    # 2. Prepare symbols and models
    # 【Option 1: Default】Batch test all checkpoints discovered in saves/ across test_data
    # 【Option 2: Custom List】engine.prepare(target_symbols, model_paths=['checkpoint_iter_10.pt', ...])
    # 【Option 3: Specific Dir】engine.prepare(target_symbols, model_dir='saves/20261017-204239-300k-')
    engine.prepare(
        target_symbols=args.symbols,
        model_paths=args.models,
        model_dir=args.model_dir,
    )

    if args.formal:
        # Formal Live Mode: Export formal_if_order_map.json
        print("\n[PPO2_rl_test] Generating latest target order map for live execution...")
        order_map_path = os.path.join(args.output_dir, "formal_if_order_map.json")
        if_order_map = engine.get_if_order_map(output_path=order_map_path)
        print(f"[PPO2_rl_test] Completed. Total symbols in order map: {len(if_order_map)}")
        return if_order_map
    else:
        # Batch Test Mode: Stream evaluate all models & symbols, export CSV orders & manifests
        print("\n[PPO2_rl_test] Starting streaming evaluation loop...")
        manifest_df = engine.run_test(output_dir=args.output_dir)
        print(f"[PPO2_rl_test] Completed. Total records in consolidated manifest: {len(manifest_df)}")
        return manifest_df


if __name__ == "__main__":
    main()
