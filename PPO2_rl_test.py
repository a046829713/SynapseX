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
    parser.add_argument(
        "--test_start_date",
        type=str,
        default=None,
        help="Optional override for evaluation start timestamp (e.g. '2026-01-01 00:00:00')",
    )
    parser.add_argument(
        "--test_end_date",
        type=str,
        default=None,
        help="Optional override for evaluation end timestamp (e.g. '2025-12-31 23:59:59')",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride interval for sampling checkpoints during screening (e.g. --stride 10)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top checkpoints to recommend for ensemble selection",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        default=False,
        help="Run in Sidecar Watcher Mode to continuously detect new .pt files and accumulate leaderboard.csv",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=30.0,
        help="Polling interval in seconds when --watch is enabled (default: 30.0)",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        default=False,
        help="Disable incremental manifest.json caching and force full re-evaluation",
    )
    return parser.parse_args()


def _print_leaderboard_summary(engine: PPO2Engine, top_k: int = 5):
    lb_df = engine.last_leaderboard_df
    summary = engine.last_selection_summary or {}
    if lb_df is None or lb_df.empty:
        return

    print("\n" + "=" * 70)
    print(f" Top-{min(top_k, len(lb_df))} Checkpoints Leaderboard Summary")
    print("=" * 70)
    display_cols = [
        "rank",
        "model_name",
        "gate_pass",
        "plateau_score",
        "iqm_ir_2x",
        "iqm_sharpe",
        "median_mdd_pct",
        "beat_bh_ratio",
        "dsr_prob",
    ]
    existing_cols = [c for c in display_cols if c in lb_df.columns]
    print(lb_df[existing_cols].head(top_k).to_string(index=False))

    selected_paths = summary.get("selected_ensemble_paths", [])
    if selected_paths:
        models_arg = " ".join(selected_paths)
        print("\n[Recommended Formal Ensemble Command]:")
        print(f"../bin/python PPO2_rl_test.py --formal --models {models_arg}")
    elif summary.get("fallback_top_paths"):
        fallback_arg = " ".join(summary["fallback_top_paths"])
        print("\n[Note: No checkpoint passed all strict hard gates. Top fallback command]:")
        print(f"../bin/python PPO2_rl_test.py --formal --models {fallback_arg}")
    print("=" * 70)


def main():
    args = parse_arguments()

    if args.formal:
        mode_str = "Formal Live Execution"
    elif args.watch:
        mode_str = f"Sidecar Watcher & Live Leaderboard Accumulation (poll={args.poll_interval}s)"
    else:
        mode_str = "Historical Backtest & Leaderboard Ranking"

    print("=" * 70)
    print(" PPO2 Dedicated Testing, Leaderboard & Order Export Environment")
    print(f" Mode: {mode_str}")
    print(f" Config: {args.config}")
    print(f" Output Directory: {args.output_dir}")
    print("=" * 70)

    # 1. Initialize PPO2Engine
    engine = PPO2Engine(
        formal=args.formal,
        config=args.config,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
    )
    if not args.formal:
        print(
            f"[PPO2_rl_test] Evaluation Window: "
            f"[{engine.test_start_date or 'START'} -> {engine.test_end_date or 'END'}]"
        )

    if args.watch and not args.formal:
        lb_df = engine.watch_and_evaluate(
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            target_symbols=args.symbols,
            poll_interval=args.poll_interval,
            stride=args.stride,
            top_k=args.top_k,
        )
        _print_leaderboard_summary(engine, top_k=args.top_k)
        return lb_df

    # 2. Prepare symbols and models
    # 【Option 1: Default】Batch test all checkpoints discovered in saves/ across test_data
    # 【Option 2: Custom List】engine.prepare(target_symbols, model_paths=['checkpoint_iter_10.pt', ...])
    # 【Option 3: Specific Dir】engine.prepare(target_symbols, model_dir='saves/20261017-204239-300k-')
    engine.prepare(
        target_symbols=args.symbols,
        model_paths=args.models,
        model_dir=args.model_dir,
        stride=args.stride,
    )

    if args.formal:
        # Formal Live Mode: Export formal_if_order_map.json
        print("\n[PPO2_rl_test] Generating latest target order map for live execution...")
        order_map_path = os.path.join(args.output_dir, "formal_if_order_map.json")
        if_order_map = engine.get_if_order_map(output_path=order_map_path)
        print(f"[PPO2_rl_test] Completed. Total symbols in order map: {len(if_order_map)}")
        return if_order_map
    else:
        # Batch Test Mode: Stream evaluate all models & symbols, export CSV orders, manifests & leaderboard.csv
        print("\n[PPO2_rl_test] Starting streaming evaluation loop...")
        manifest_df = engine.run_test(
            output_dir=args.output_dir,
            top_k=args.top_k,
            use_cache=not args.no_cache,
        )
        _print_leaderboard_summary(engine, top_k=args.top_k)
        print(f"[PPO2_rl_test] Completed. Total records in consolidated manifest: {len(manifest_df)}")
        return manifest_df


if __name__ == "__main__":
    main()
