"""Adaptive-EAGLE demo: speculative decoding with SMW-adapted draft head.

Loads a draft/target model pair and runs the full speculative decoding loop
with online adaptation. Prints generated text, acceptance stats, and timing.

Usage:
    python run_demo.py
    python run_demo.py --draft gpt2 --target gpt2-xl --max-tokens 100
    python run_demo.py --prompt "Once upon a time" --device cpu
    python run_demo.py --sim-only --show-hybrid-engine
"""
from __future__ import annotations

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive-EAGLE speculative decoding demo")
    p.add_argument("--draft", default="gpt2", help="Draft model name (default: gpt2)")
    p.add_argument("--target", default="gpt2-xl", help="Target model name (default: gpt2-xl)")
    p.add_argument("--prompt", default="The future of artificial intelligence", help="Input prompt")
    p.add_argument("--max-tokens", type=int, default=100, help="Max new tokens to generate")
    p.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    p.add_argument("--draft-length", type=int, default=5, help="Tokens to draft per round")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--do-sample", action="store_true", help="Sample from draft distribution instead of greedy argmax")
    p.add_argument("--skip-simulation", action="store_true", help="Skip pipeline simulation report")
    p.add_argument("--sim-only", action="store_true", help="Run only simulation reports (no model loading)")
    p.add_argument("--show-inference-engine", action="store_true", help="Show inference engine tier comparison report")
    p.add_argument("--show-hybrid-engine", action="store_true", help="Show hybrid CPU+GPU inference engine report")
    p.add_argument("--show-optimizations", action="store_true", help="Show I/O optimization comparison report")
    p.add_argument("--export-model", nargs=2, metavar=("MODEL", "OUTPUT_DIR"),
                   help="Export a HuggingFace model to streaming format")
    p.add_argument("--streaming", metavar="STORE_DIR",
                   help="Use streaming target model from exported weight store")
    return p.parse_args()


def _run_simulation_reports(args):
    """Run physics-based simulation reports without loading any models."""
    if not args.skip_simulation:
        from neuralbyte.spec_decode.layer_stream import compare_modes, print_layer_stream_report
        from neuralbyte.spec_decode.pipeline import PipelineConfig, print_pipeline_report, simulate_pipeline

        print("=" * 60)
        print("  PIPELINE SIMULATION (physics-based estimate)")
        print("=" * 60)
        print()

        sim_result = simulate_pipeline()
        report = print_pipeline_report(sim_result)
        print(report)
        print()

        print()
        comparison = compare_modes()
        ls_report = print_layer_stream_report(comparison)
        print(ls_report)

    if args.show_inference_engine:
        from neuralbyte.spec_decode.inference_engine import (
            compare_tiers,
            print_inference_engine_report,
        )

        print()
        tier_comparison = compare_tiers()
        engine_report = print_inference_engine_report(tier_comparison)
        print(engine_report)

    if args.show_hybrid_engine:
        from neuralbyte.spec_decode.hybrid_engine import (
            print_hybrid_engine_report,
            simulate_hybrid_engine,
        )

        print()
        hybrid_result = simulate_hybrid_engine()
        hybrid_report = print_hybrid_engine_report(hybrid_result)
        print(hybrid_report)

    if args.show_optimizations:
        from neuralbyte.spec_decode.hybrid_engine import print_optimization_report

        print()
        print(print_optimization_report())


def main():
    args = parse_args()

    print("=" * 60)
    print("  ADAPTIVE-EAGLE: Speculative Decoding Demo")
    print("=" * 60)
    print()

    # --- Export mode ---
    if args.export_model:
        from neuralbyte.spec_decode.weight_store import WeightStore

        model_name, output_dir = args.export_model
        print(f"Exporting {model_name} to {output_dir}...")
        WeightStore.export(model_name, output_dir)
        print(f"Export complete. Use --streaming {output_dir} to run inference.")
        if not args.streaming and not args.sim_only:
            return

    # --- Simulation-only mode: skip all model loading ---
    if args.sim_only:
        _run_simulation_reports(args)
        return

    # --- Full mode: load models and generate ---
    import torch

    from neuralbyte.spec_decode.config import OSDConfig
    from neuralbyte.spec_decode.engine import SpeculativeEngine

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Device: {device} ({gpu_name}, {vram_gb:.1f} GB VRAM)")
    else:
        print(f"Device: {device}")
    print()

    config = OSDConfig(
        draft_length=args.draft_length,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )

    if args.streaming:
        print(f"Loading draft model:  {args.draft}")
        print(f"Streaming target:     {args.streaming}")
        print()

        engine = SpeculativeEngine.from_streaming(
            store_path=args.streaming,
            draft_model_name=args.draft,
            config=config,
            device=device,
        )
    else:
        print(f"Loading draft model:  {args.draft}")
        print(f"Loading target model: {args.target}")
        print()

        try:
            engine = SpeculativeEngine.from_models(
                draft_model_name=args.draft,
                target_model_name=args.target,
                config=config,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory — falling back to CPU...")
            device = "cpu"
            torch.cuda.empty_cache()
            engine = SpeculativeEngine.from_models(
                draft_model_name=args.draft,
                target_model_name=args.target,
                config=config,
                device=device,
            )

    head_stats = engine.draft_head.stats()
    print(f"Draft head initialized:")
    print(f"  Feature dim: {head_stats['feature_dim']}")
    print(f"  Vocab size:  {head_stats['vocab_size']}")
    print(f"  Trace:       {head_stats['trace']:.2f}")
    print()

    # --- Generate ---
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Generating {args.max_tokens} tokens (draft_length={args.draft_length})...")
    print()

    result = engine.generate(args.prompt, max_new_tokens=args.max_tokens)

    # --- Results ---
    print("-" * 60)
    print("Generated text:")
    print("-" * 60)
    print(result.text)
    print("-" * 60)
    print()

    print("Statistics:")
    print(f"  Tokens generated:  {len(result.token_ids)}")
    print(f"  Rounds:            {result.n_rounds}")
    print(f"  Total drafted:     {result.total_drafted}")
    print(f"  Total accepted:    {result.total_accepted}")
    print(f"  Acceptance rate:   {result.acceptance_rate:.1%}")
    print(f"  Updates applied:   {result.updates_applied}")
    print(f"  Wall time:         {result.wall_time_s:.2f}s")
    print(f"  Tokens/sec:        {result.tokens_per_second:.1f}")
    print()

    tracker_dict = result.tracker.to_dict()
    gain = tracker_dict["adaptation_gain"]
    if gain is not None:
        direction = "improving" if gain > 0 else "declining" if gain < 0 else "flat"
        print(f"  Adaptation gain:   {gain:+.2%} ({direction})")
    else:
        print(f"  Adaptation gain:   N/A (< 8 rounds)")
    print()

    rates = result.acceptance_curve
    if rates:
        print("Per-round acceptance:")
        bar_width = 30
        for i, rate in enumerate(rates):
            bar = "#" * int(rate * bar_width)
            print(f"  Round {i+1:3d}: {rate:5.1%} |{bar:<{bar_width}}|")
        print()

    post_stats = engine.draft_head.stats()
    print(f"Draft head after adaptation:")
    print(f"  Updates:     {post_stats['update_count']}")
    print(f"  Window fill: {post_stats['window_fill']}/{post_stats['window_capacity']}")
    print(f"  Trace:       {post_stats['trace']:.2f}")
    print()

    # --- Simulation reports (after generation) ---
    _run_simulation_reports(args)


if __name__ == "__main__":
    main()
