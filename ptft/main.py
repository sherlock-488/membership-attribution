#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
import re
import os

from util import SAVE_PATH


def run_cmd(cmd, capture=False):
    """小工具：跑子进程，默认实时打印，必要时报错退出。"""
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
    从 evaluate_attacker.py 的输出里解析 rec_value=tensor([...]) 这一行，
    返回前三个类的 recall（float）。
    """
    rec_line = None
    for line in stdout.splitlines():
        if "rec_value=" in line:
            rec_line = line
            break

    if rec_line is None:
        print("[WARN] 没找到 rec_value= 这一行，无法解析论文风格的 MA 数字。")
        return None

    # 形如：rec_value=tensor([0.9956, 0.9984, 0.9992, 0.9936])
    m = re.search(r"rec_value=tensor\(\[([0-9eE\.\,\s\-]+)\]\)", rec_line)
    if not m:
        print("[WARN] rec_value 行解析失败：", rec_line)
        return None

    nums_str = m.group(1)
    nums = [float(x.strip()) for x in nums_str.split(",") if x.strip()]

    if len(nums) < 3:
        print("[WARN] rec_value 里元素不足 3 个：", nums)
        return None

    return nums[0], nums[1], nums[2]


def main():
    parser = argparse.ArgumentParser(
        description="一条命令跑完整 MA 实验（finetune -> feature -> attacker -> evaluate）"
    )
    parser.add_argument("--model", default="llama", help="util.py 里 model_dict 的 key，比如 llama / gpt-neo")
    parser.add_argument(
        "--dataname",
        default="agnews",
        help="数据集名称：agnews / onion / oloma / mimir / wikimia"
    )
    parser.add_argument(
        "--attack",
        default="ma_diff_w",
        help=(
            "攻击特征名：ma / ma_w / ma_diff_w 等；"
            "若为 cascade_mia，则走级联 MI/MIU 流程"
        )
    )
    parser.add_argument("--pt_access", default="open", help="预训练模型访问方式：open / close")
    parser.add_argument("--ft_access", default="open", help="微调模型访问方式：open / close")
    parser.add_argument("--feature_dim", type=int, default=32, help="特征维度 feature_dim")

    # 新增：shadow 模型可以跟 target 模型不同
    parser.add_argument(
        "--shadow_model",
        default=None,
        help="shadow 模型使用的 base（util.model_dict 的 key），不填则默认等于 --model"
    )

    # 控制 token feature 的策略
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
        help=(
            "token 特征模式：\n"
            "  topk                     -> entr + m_entr + topK 概率（K=feature_dim-2）\n"
            "  top_bottom               -> entr + m_entr + topK/2 + bottomK/2\n"
            "  pws_ent_cum              -> [p_true, w, s_t, entr, m_entr, C_t]\n"
            "  pws_ent_cum_margin       -> 上一项 + margin_ft + delta_margin\n"
            "  pws_ent_cum_norm         -> 上一项 + C_t_norm\n"
            "  pws_ent_cum_posneg       -> 上一项 + C_t_pos + C_t_neg\n"
            "  pws_ent_cum_bidir        -> [p_true, w, s_t, entr, m_entr, C_prefix, C_suffix]\n"
            "  pws_ent_cum_global       -> 上一项 + S_total"
        )
    )

    # 有些步骤很慢，可以加开关跳过（比如已经 finetune 过）
    parser.add_argument("--skip_finetune", action="store_true", help="已训练好 target/shadow 时跳过 finetune")
    parser.add_argument("--skip_feature", action="store_true", help="已存在 feature .pt 时跳过 extract_feature")
    parser.add_argument("--skip_train_attacker", action="store_true", help="已训练好攻击模型时跳过 train_attacker")

    # 新增：多 seed 实验
    parser.add_argument(
        "--seeds",
        type=str,
        default="3407",
        help="逗号分隔的一组 random seeds，用于重复训练 / 评估 attacker，比如 '1,2,3,4,5'"
    )

    args = parser.parse_args()

    # shadow_model 默认等于 target model
    if args.shadow_model is None:
        args.shadow_model = args.model

    py = sys.executable  # 用当前 venv 的 python
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"[main] seeds = {seed_list}")
    print(f"[main] target_model = {args.model}, shadow_model = {args.shadow_model}")

    # ========== 新增：cascade MIA 分支 ==========
    # 当 attack == 'cascade_mia' 时，走“pretrain+posttrain -> 直接级联 MI/MIU 评估”的流程
    if args.attack == "cascade_mia":
        # 1) 仍然需要 finetune target/shadow（用于构造 M2 = base+LoRA(second)）
        if not args.skip_finetune:
            print("\n===== [cascade] Step 1: Finetune target & shadow =====")
            # target: 用 args.model
            run_cmd([py, "finetune.py", args.model,        args.dataname, "target"])
            # shadow: 用 args.shadow_model
            run_cmd([py, "finetune.py", args.shadow_model, args.dataname, "shadow"])
        else:
            print("\n[SKIP] 跳过 finetune 步骤（假设 target/shadow 的 LoRA 权重已存在）")

        # 2) 直接调用 cascade_mia.py 进行 MI/MIU 级联评估
        print("\n===== [cascade] Step 2: Run cascade_mia.py =====")
        # 这里把 target_model_key / dataname / shadow_model_key 都传进去
        run_cmd([py, "cascade_mia.py", args.model, args.dataname, args.shadow_model])
        return
    # ========== cascade 分支结束 ==========

    # 结果汇总文件：<SAVE_PATH>/seed_results/<dataname>.txt
    summary_dir = os.path.join(SAVE_PATH, "seed_results")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{args.dataname}.txt")
    print(f"[main] summary file: {summary_path}")

    # 1) finetune target & shadow（只跑一次，后面所有 seed 复用）
    if not args.skip_finetune:
        print("\n===== Step 1: Finetune target & shadow =====")
        run_cmd([py, "finetune.py", args.model,        args.dataname, "target"])
        run_cmd([py, "finetune.py", args.shadow_model, args.dataname, "shadow"])
    else:
        print("\n[SKIP] 跳过 finetune 步骤（假设权重已存在）")

    # 2) 提取 shadow & target 特征（只跑一次）
    if not args.skip_feature:
        print("\n===== Step 2: Extract membership features (shadow + target) =====")
        run_cmd([
            py, "extract_feature.py",
            args.attack,
            args.shadow_model,  # shadow 这边用 shadow_model
            args.dataname,
            "shadow",
            args.pt_access,
            args.ft_access,
            str(args.feature_dim),
            args.token_feature_mode,
        ])
        run_cmd([
            py, "extract_feature.py",
            args.attack,
            args.model,         # target 这边用 target_model
            args.dataname,
            "target",
            args.pt_access,
            args.ft_access,
            str(args.feature_dim),
            args.token_feature_mode,
        ])
    else:
        print("\n[SKIP] 跳过 feature 提取（假设 .pt 已存在）")

    # 3+4) 对每个 seed 训练 + 评估 attacker，并把结果写进同一个 txt
    for seed in seed_list:
        # 3) 训练攻击模型（攻击模型本身是 shadow world 上训练的）
        if not args.skip_train_attacker:
            print(f"\n===== Step 3: Train attacker (seed={seed}) =====")
            run_cmd([
                py, "train_attacker.py",
                args.attack,
                args.model,         # target_model
                args.shadow_model,  # shadow_model
                args.dataname,
                args.pt_access,
                args.ft_access,
                str(args.feature_dim),
                str(seed),
            ])
        else:
            print(f"\n[SKIP] seed={seed} 跳过攻击模型训练（假设该 seed 下的 .pth 已存在）")

        # 4) 评估 & 解析论文风格指标
        print(f"\n===== Step 4: Evaluate attacker (seed={seed}) =====")
        eval_stdout = run_cmd([
            py, "evaluate_attacker.py",
            args.attack,
            args.model,         # target_model
            args.shadow_model,  # shadow_model
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
            print(
                "MA {:.2f}% {:.2f}% {:.2f}%".format(
                    r0 * 100.0, r1 * 100.0, r2 * 100.0
                )
            )
        else:
            print("\n[WARN] 没能解析出论文风格的 MA 数字，请手动看 evaluate_attacker.py 的输出。")

        # 把完整输出 + 解析后的 MA 追加写入 summary 文本
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
