#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
import re
import os

HERE = os.path.dirname(__file__)
MA_DIR = os.path.join(HERE, "MA")
CASCADE_DIR = os.path.join(HERE, "CascadedMA")
sys.path.append(MA_DIR)
from pt_ft_util import SAVE_PATH  # noqa: E402


def run_cmd(cmd, capture=False):
    """Run a subprocess; stream output by default."""
    print(f"\n[RUN] {' '.join(cmd)}")
    if capture:
        res = subprocess.run(cmd, text=True, capture_output=True)
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        if res.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        return res.stdout
    else:
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        return None


def parse_rec_value_from_output(stdout):
    """
    Parse rec_value=tensor([...]) line from evaluate_attacker output.
    Returns first three recall values as floats. Accepts both tensor(...) and numpy array style.
    """
    rec_line = None
    for line in stdout.splitlines():
        if "rec_value=" in line:
            rec_line = line
            break

    if rec_line is None:
        print("[WARN] rec_value line not found; cannot parse paper-style MA")
        return None

    # match tensor([...]) or array([...])
    match = re.search(r"rec_value=(?:tensor|array)\(\[([0-9eE\.\,\s\-]+)\]", rec_line)
    if not match:
        print("[WARN] failed to parse rec_value line:", rec_line)
        return None

    nums_str = match.group(1)
    nums = [float(x.strip()) for x in nums_str.split(",") if x.strip()]
    if len(nums) < 3:
        print("[WARN] rec_value has fewer than 3 elements:", nums)
        return None
    return nums[0], nums[1], nums[2]


def main():
    parser = argparse.ArgumentParser(
        description="One-shot MA pipeline: finetune -> extract features -> train attacker -> evaluate"
    )
    parser.add_argument("--model", default="llama", help="key in model_dict, e.g., llama / gpt-neo")
    parser.add_argument(
        "--dataname",
        default="agnews",
        help="dataset name: agnews / onion / oloma / mimir / wikimia"
    )
    parser.add_argument(
        "--attack",
        default="ma_diff_w",
        help="attack feature name (ma / ma_w / ma_diff_w / cascade_mia)"
    )
    parser.add_argument("--pt_access", default="open", help="pretrain access: open / close")
    parser.add_argument("--ft_access", default="open", help="finetune access: open / close")
    parser.add_argument("--feature_dim", type=int, default=32, help="feature_dim")
    parser.add_argument(
        "--shadow_model",
        default=None,
        help="shadow base model (model_dict key); default same as --model"
    )
    parser.add_argument(
        "--token_feature_mode",
        default="topk",
        choices=[
            "topk",
            "top_bottom",
            "pws_ent_cum",
            "pws_ent_cum_margin",
            "pws_ent_cum_norm",
            "pws_ent_cum_posneg",
            "pws_ent_cum_bidir",
            "pws_ent_cum_global",
        ],
        help="token feature strategy"
    )
    parser.add_argument("--skip_finetune", action="store_true", help="skip finetune if weights exist")
    parser.add_argument("--skip_feature", action="store_true", help="skip feature extraction if .pt exists")
    parser.add_argument("--skip_train_attacker", action="store_true", help="skip attacker training if .pth exists")
    parser.add_argument(
        "--seeds",
        type=str,
        default="3407",
        help="comma-separated seeds for attacker training/eval, e.g., '1,2,3'"
    )

    args = parser.parse_args()
    if args.shadow_model is None:
        args.shadow_model = args.model

    py = sys.executable
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"[main] seeds = {seed_list}")
    print(f"[main] target_model = {args.model}, shadow_model = {args.shadow_model}")

    finetune_script = os.path.join("MA", "finetune.py")
    extract_script = os.path.join("MA", "extract_pt_ft.py")
    train_script = os.path.join("MA", "train_attacker.py")
    eval_script = os.path.join("MA", "evaluate_attacker.py")
    cascade_script = os.path.join("CascadedMA", "cascade_mia_pt_ft.py")

    # Cascade branch
    if args.attack == "cascade_mia":
        if not args.skip_finetune:
            print("\n===== [cascade] Step 1: Finetune target & shadow =====")
            run_cmd([py, finetune_script, args.model,        args.dataname, "target"])
            run_cmd([py, finetune_script, args.shadow_model, args.dataname, "shadow"])
        else:
            print("\n[SKIP] skip finetune (assume target/shadow LoRA already present)")

        print("\n===== [cascade] Step 2: Run cascade_mia =====")
        run_cmd([py, cascade_script, args.model, args.dataname, args.shadow_model])
        return

    summary_dir = os.path.join(SAVE_PATH, "seed_results")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{args.dataname}.txt")
    print(f"[main] summary file: {summary_path}")

    # Step 1: finetune
    if not args.skip_finetune:
        print("\n===== Step 1: Finetune target & shadow =====")
        run_cmd([py, finetune_script, args.model,        args.dataname, "target"])
        run_cmd([py, finetune_script, args.shadow_model, args.dataname, "shadow"])
    else:
        print("\n[SKIP] skip finetune (assume weights exist)")

    # Step 2: feature extraction
    if not args.skip_feature:
        print("\n===== Step 2: Extract membership features (shadow + target) =====")
        run_cmd([
            py, extract_script,
            args.attack,
            args.shadow_model,
            args.dataname,
            "shadow",
            args.pt_access,
            args.ft_access,
            str(args.feature_dim),
            args.token_feature_mode,
        ])
        run_cmd([
            py, extract_script,
            args.attack,
            args.model,
            args.dataname,
            "target",
            args.pt_access,
            args.ft_access,
            str(args.feature_dim),
            args.token_feature_mode,
        ])
    else:
        print("\n[SKIP] skip feature extraction (assume .pt exists)")

    # Step 3/4: train + eval attacker per seed
    for seed in seed_list:
        if not args.skip_train_attacker:
            print(f"\n===== Step 3: Train attacker (seed={seed}) =====")
            run_cmd([
                py, train_script,
                args.attack,
                args.model,
                args.shadow_model,
                args.dataname,
                args.pt_access,
                args.ft_access,
                str(args.feature_dim),
                str(seed),
            ])
        else:
            print(f"\n[SKIP] seed={seed} skip attacker training (assume .pth exists)")

        print(f"\n===== Step 4: Evaluate attacker (seed={seed}) =====")
        eval_stdout = run_cmd([
            py, eval_script,
            args.attack,
            args.model,
            args.shadow_model,
            args.dataname,
            args.pt_access,
            args.ft_access,
            str(args.feature_dim),
            str(seed),
        ], capture=True)

        rec_tuple = parse_rec_value_from_output(eval_stdout)
        if rec_tuple is not None:
            r0, r1, r2 = rec_tuple
            print("\n===== Paper-style MA (per-owner recall) =====")
            print("MA {:.2f}% {:.2f}% {:.2f}%".format(r0 * 100.0, r1 * 100.0, r2 * 100.0))
        else:
            print("\n[WARN] cannot parse paper-style MA; please check evaluate output")

        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(
                f"===== dataname={args.dataname}, model={args.model}, shadow_model={args.shadow_model}, "
                f"attack={args.attack}, seed={seed} =====\n"
            )
            f.write(eval_stdout)
            if rec_tuple is not None:
                f.write(
                    "\nParsed MA: {:.2f}% {:.2f}% {:.2f}%\n".format(
                        r0 * 100.0, r1 * 100.0, r2 * 100.0
                    )
                )
            f.write("\n\n")


if __name__ == "__main__":
    main()
