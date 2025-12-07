# cascade_mia.py
import os
import sys
import zlib
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, f1_score

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "MA"))
from pt_ft_util import SAVE_PATH, model_dict, load_eval_data, get_last_prob  # noqa: E402

DEVICE = "cuda"

# 用环境变量控制 MI 方法：
#   CASCADE_MI_ATTACK = loss | zlib | ref | min-k++ | recall | con-recall
MI_METHOD_RAW = os.getenv("CASCADE_MI_ATTACK", "loss").lower()

if MI_METHOD_RAW in ["min-k++", "minkpp", "minkplusplus", "min_k++", "min_kpp"]:
    MI_METHOD = "minkpp"
elif MI_METHOD_RAW in ["conrecall", "con-recall", "con_recall"]:
    MI_METHOD = "con_recall"
else:
    MI_METHOD = MI_METHOD_RAW

VALID_MI = {"loss", "zlib", "ref", "minkpp", "recall", "con_recall"}
if MI_METHOD not in VALID_MI:
    raise ValueError(f"[cascade] CASCADE_MI_ATTACK={MI_METHOD_RAW!r} 不在支持列表 {VALID_MI}")

# Min-k++ 的 k 百分比（默认 5%）
MIN_K_PERCENT = float(os.getenv("CASCADE_MIN_K_PERCENT", "0.05"))

# ReCaLL / Con-ReCaLL 相关超参
RECALL_NUM_SHOTS = int(os.getenv("CASCADE_RECALL_SHOTS", "7"))
CONRECALL_GAMMA = float(os.getenv("CONRECALL_GAMMA", "0.5"))

# ReCaLL / Con-ReCaLL 用到的全局 prefix：
# 为了让 shadow / target 各自用自己的 prefix，这里按 training_mode 存一份。
# key: "shadow" / "target"
_RECALL_PREFIX_NONMEM = {}
_RECALL_PREFIX_MEM = {}

# === ref attack 的参考模型固定为 stablelm-base-alpha-3b-v2（不用 model_dict） ===
REF_MODEL_NAME = "stabilityai/stablelm-base-alpha-3b-v2"

_REF_MODEL = None
_REF_TOKENIZER = None


def get_ref_model():
    """
    参考模型，用于 ref attack:
      f_Ref(X;M) = L(X;M) - L(X;M_ref)
    membership score 用 (L_ref - L) => 越大越像 member
    这里固定用 HF 模型 'stabilityai/stablelm-base-alpha-3b-v2'。
    """
    global _REF_MODEL, _REF_TOKENIZER
    if _REF_MODEL is not None:
        return _REF_MODEL, _REF_TOKENIZER

    print(f"[cascade] loading REF model: {REF_MODEL_NAME}")
    quant = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        REF_MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(REF_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    _REF_MODEL, _REF_TOKENIZER = model, tokenizer
    return _REF_MODEL, _REF_TOKENIZER


@torch.no_grad()
def sentence_loss(model, tokenizer, texts, max_length=128, batch_size=8):
    """
    对一批文本算 avg negative log-likelihood（交叉熵），返回 shape [N]。
    这是用在：
      - MIU: loss_M1 - loss_M2
      - LOSS MI: MI = -loss_M1（越大越像 member）
      - ZLIB, REF: 都要用到 L(X; M)
    """
    model.eval()
    all_loss = []
    for i in tqdm(range(0, len(texts), batch_size), desc="compute loss"):
        batch = texts[i:i + batch_size]
        toks = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).to(DEVICE)

        input_ids = toks.input_ids
        attn = toks.attention_mask

        outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits[:, :-1, :]        # [B, L-1, V]
        labels = input_ids[:, 1:]                 # [B, L-1]

        pad_id = tokenizer.pad_token_id
        mask = (labels != pad_id) & (attn[:, 1:] == 1)   # [B, L-1]

        log_probs = torch.log_softmax(logits, dim=-1)    # [B, L-1, V]
        token_ll = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        token_ll = token_ll * mask

        sum_ll = token_ll.sum(dim=1)                     # [B]
        count = mask.sum(dim=1).clamp_min(1)             # [B]
        avg_nll = -sum_ll / count                        # [B]
        all_loss.append(avg_nll.cpu().numpy())

    return np.concatenate(all_loss, axis=0)              # [N]


@torch.no_grad()
def conditional_sentence_loss(model, tokenizer, texts, prefix, max_length=128, batch_size=8):
    """
    计算 avg NLL(x | prefix)：
      - prefix: 单个前缀字符串（所有样本共用）
      - 对每个样本构造: [prefix_tokens] + [x_tokens]，只在 x 的 token 上求 loss
      - 返回 shape [N] 的平均 NLL

    只在 ReCaLL / Con-ReCaLL 里使用。
    """
    model.eval()

    # 先 tokenize prefix 一次（不加 special tokens）
    pref_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    # 给 x 留点位置，避免 prefix 太长挤掉所有 x token
    if len(pref_ids) > max_length - 4:
        pref_ids = pref_ids[: max_length - 4]
    pref_len = len(pref_ids)

    pad_id = tokenizer.pad_token_id
    all_loss = []

    for i in tqdm(range(0, len(texts), batch_size), desc="compute cond loss"):
        batch = texts[i:i + batch_size]
        batch_input_ids = []
        batch_attn = []
        batch_pref_lens = []

        for t in batch:
            x_ids = tokenizer(t, add_special_tokens=False)["input_ids"]
            # 只保留能塞进 max_length 的部分
            max_x_len = max_length - pref_len
            if max_x_len <= 0:
                x_ids = []
            else:
                x_ids = x_ids[:max_x_len]

            ids = pref_ids + x_ids
            attn = [1] * len(ids)

            # padding
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids += [pad_id] * pad_len
                attn += [0] * pad_len

            batch_input_ids.append(ids)
            batch_attn.append(attn)
            batch_pref_lens.append(pref_len)

        input_ids = torch.tensor(batch_input_ids, device=DEVICE)
        attn = torch.tensor(batch_attn, device=DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits[:, :-1, :]  # [B, L-1, V]
        labels = input_ids[:, 1:]           # [B, L-1]

        base_mask = (labels != pad_id) & (attn[:, 1:] == 1)  # 有效 token

        # 只保留 x 部分的 label 位置：
        # token 索引: 0..L-1
        # prefix token: 0..pref_len-1
        # x token:      pref_len..L-1
        # label 索引 j 对应 token 索引 j+1 -> 要求 j+1 >= pref_len => j >= pref_len-1
        B, Lm1 = labels.shape
        mask_x = torch.zeros_like(base_mask, dtype=torch.bool)
        for b in range(B):
            Lp = min(batch_pref_lens[b], Lm1 + 1)
            start = max(Lp - 1, 0)
            if start < Lm1:
                mask_x[b, start:] = base_mask[b, start:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        token_ll = token_ll * mask_x

        sum_ll = token_ll.sum(dim=1)
        count = mask_x.sum(dim=1).clamp_min(1)
        avg_nll = -sum_ll / count
        all_loss.append(avg_nll.cpu().numpy())

    return np.concatenate(all_loss, axis=0)


def build_models(model_key, dataname, training_mode):
    """
    M1 = base model（HF 模型，不带 LoRA，8bit）
    M2 = base + LoRA(second)（finetune.py 训练出来的 second，8bit）

    注意：model_key 是 util.model_dict 的 key（如 'gpt-neo' / 'llama' / ...）
    """
    base_name = model_dict[model_key]
    quant = BitsAndBytesConfig(load_in_8bit=True)

    # 注意：8bit + device_map="auto" 的模型不能再 .to(...)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant,
    )
    base_model.eval()

    base_for_second = AutoModelForCausalLM.from_pretrained(
        base_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant,
    )
    base_for_second.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model_id_second = os.path.join(
        SAVE_PATH, f"weights/{base_name}/{dataname}_{training_mode}_second"
    )
    if not os.path.isdir(model_id_second):
        raise RuntimeError(
            f"找不到 second LoRA 权重: {model_id_second}，"
            f"请先运行: python finetune.py {model_key} {dataname} {training_mode}"
        )

    second_model = PeftModel.from_pretrained(
        base_for_second,
        model_id=model_id_second,
        adapter_name="second"
    )
    second_model.eval()
    second_model.base_model.requires_grad_(False)

    return base_model, second_model, tokenizer


# ================== 辅助：ZLIB 复杂度 ==================

def zlib_complexity(text: str) -> float:
    """
    zlib(X): 用压缩长度 / 原始长度 表示文本复杂度。
    越大表示文本越复杂。
    """
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) == 0:
        return 1.0
    comp = zlib.compress(raw)
    return float(len(comp)) / float(len(raw))


# ================== Min-k++ 实现 ==================

@torch.no_grad()
def minkpp_scores(model, tokenizer, texts, max_length=128, batch_size=8, k_percent=0.2):
    """
    Min-K%++ attack:
      1. 计算每个 token 的 loss L_t(x_t) = -log p(x_t | prefix)
      2. 对每个位置 t，计算 over vocab 的期望 µ_<t 和 std σ_<t（对 L_t(v)）
      3. 归一化：  tilde_L_t = (L_t(x_t) - µ_<t) / σ_<t
      4. 按原始 L_t 选出 top k% loss 的 token 集合 min-k(X)
      5. f_MinKpp(X) = (1/|min-k|) * sum_{t in min-k} tilde_L_t
         membership score = - f_MinKpp  （越小越像 member -> 取负号）
    返回 membership score, numpy 数组 [N]，越大越像 member。
    """
    model.eval()
    scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"min-k++ (k={k_percent:.3f})"):
        batch_texts = texts[i:i + batch_size]
        toks = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).to(DEVICE)

        input_ids = toks.input_ids                # [B, L]
        attn = toks.attention_mask                # [B, L]
        pad_id = tokenizer.pad_token_id

        # 利用 util.get_last_prob：softmax(logits)[..., :-1, :]
        probs = get_last_prob(model, toks)        # [B, L-1, V]
        log_probs = torch.log(probs.clamp_min(1e-30))  # [B, L-1, V]
        L_all = -log_probs                               # per-vocab loss

        # µ_<t, σ_<t over vocab
        mu = L_all.mean(dim=-1)                 # [B, L-1]
        sigma = L_all.std(dim=-1)               # [B, L-1]
        sigma = sigma.clamp_min(1e-6)

        labels = input_ids[:, 1:]               # [B, L-1]
        valid_mask = (labels != pad_id) & (attn[:, 1:] == 1)   # [B, L-1]

        # 真实 token 的 loss L_t(x_t)
        L_true = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

        # 归一化 tilde_L
        tilde = (L_true - mu) / sigma           # [B, L-1]

        B_cur = L_true.size(0)
        for b in range(B_cur):
            m = valid_mask[b]
            L_b = L_true[b][m]          # [T]
            tilde_b = tilde[b][m]       # [T]

            if L_b.numel() == 0:
                scores.append(0.0)
                continue

            k = max(1, int(L_b.numel() * k_percent))
            # 选 top-k 最大的 loss token 作为 min-k(X)
            top_vals, top_idx = torch.topk(L_b, k, largest=True, sorted=False)
            tilde_top = tilde_b[top_idx]

            f_minkpp = tilde_top.mean().item()
            membership_score = -f_minkpp
            scores.append(membership_score)

    return np.asarray(scores, dtype=np.float32)


# ================== 阈值选择 & MI/MIU 计算 ==================

def choose_threshold(scores, labels, greater_is_one=True):
    """
    一维阈值选择：用 ROC + Youden's J (TPR-FPR) 选一个 tau。
    scores: np.array [N]  —— 这里我们约定“越大越像 label=1（member）”
    labels: np.array [N] in {0,1}
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    if not greater_is_one:
        scores = -scores

    if len(np.unique(labels)) < 2:
        return scores.mean(), 0.0, 0.0

    fpr, tpr, thres = roc_curve(labels, scores)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thres[idx]), float(fpr[idx]), float(tpr[idx])


def compute_scores_for_mode(model_key,
                            dataname,
                            training_mode,
                            max_length=128,
                            shadow_model_key=None):
    """
    对某个 mode（shadow / target）的 [Y,B,Z,N] 全部样本计算 MI / MIU。

    model_key: target 模型的 key（util.model_dict 的 key）
    shadow_model_key: shadow 模型的 key；若为 None，则默认与 model_key 相同。

    约定：
      - training_mode='target' 时，一定用 model_key
      - training_mode='shadow' 时，若给了 shadow_model_key，则用它对应的 base（例如 gpt-neo）
    """
    # 选本次 mode 使用哪个 base key
    if training_mode == "shadow" and shadow_model_key is not None:
        used_key = shadow_model_key
    else:
        used_key = model_key

    print(f"[cascade] compute_scores_for_mode: mode={training_mode}, base_key={used_key}")

    dataset = load_eval_data(dataname, training_mode)
    texts = list(dataset["text"])
    N = len(texts)
    assert N % 4 == 0, "期待 load_eval_data 返回 [Y,B,Z,N] 四等份"

    group_size = N // 4
    labels_class = np.repeat(np.arange(4), group_size)

    # 先算 MIU 用到的 loss_M1, loss_M2（用 LoRA 流水线里的 M1/M2）
    M1, M2, tokenizer = build_models(used_key, dataname, training_mode)

    print(f"[cascade] computing loss on M1 (pretrain) for mode={training_mode}")
    loss_M1 = sentence_loss(M1, tokenizer, texts, max_length=max_length)

    print(f"[cascade] computing loss on M2 (posttrain) for mode={training_mode}")
    loss_M2 = sentence_loss(M2, tokenizer, texts, max_length=max_length)

    # MIU: ScoreDiff (M1 -> M2)，MIU 越大表示 M2 比 M1 好得越多 => 更像 update 中被训练
    miu_scores = loss_M1 - loss_M2

    # === MI: 根据 MI_METHOD 选择 ===
    method = MI_METHOD

    if method == "loss":
        # 完全用本地 sentence_loss，当成 LOSS attack
        print("[cascade] MI method = LOSS")
        mi_scores = -loss_M1

    elif method == "zlib":
        print("[cascade] MI method = ZLIB")
        z_vals = np.array([zlib_complexity(t) for t in texts], dtype=np.float64)
        # f_Zlib = L / zlib，membership_score = - f_Zlib
        f_zlib = loss_M1 / z_vals
        mi_scores = -f_zlib

    elif method == "ref":
        print("[cascade] MI method = REF, ref_model={REF_MODEL_NAME}")
        ref_model, ref_tok = get_ref_model()
        print(f"[cascade] computing loss on REF model for mode={training_mode}")
        loss_ref = sentence_loss(ref_model, ref_tok, texts, max_length=max_length)
        # f_Ref = L - L_ref，membership_score = L_ref - L
        mi_scores = loss_ref - loss_M1

    elif method == "minkpp":
        print(f"[cascade] MI method = Min-K++ (k={MIN_K_PERCENT:.3f})")
        mi_scores = minkpp_scores(
            M1,
            tokenizer,
            texts,
            max_length=max_length,
            batch_size=8,
            k_percent=MIN_K_PERCENT,
        )

    elif method in {"recall", "con_recall"}:
        # ===== ReCaLL / Con-ReCaLL =====
        if dataname not in ["oloma", "mimir", "wikimia"]:
            raise ValueError(
                f"[cascade] MI_METHOD={method} 目前只在 OLOMA / MIMIR / WIKIMIA 上实现，"
                f"对 dataname={dataname} 未定义。"
            )

        global _RECALL_PREFIX_NONMEM, _RECALL_PREFIX_MEM

        # texts: [Y,B,Z,N] 各 group_size 个
        y_texts = texts[0 * group_size:1 * group_size]
        b_texts = texts[1 * group_size:2 * group_size]
        z_texts = texts[2 * group_size:3 * group_size]
        n_texts = texts[3 * group_size:4 * group_size]

        mem_pool = y_texts + b_texts      # pretrain member: Y,B
        nonmem_pool = z_texts + n_texts   # pretrain non-member: Z,N

        # 每个 training_mode ("shadow"/"target") 各自构建一套 prefix，一次后复用
        rng = np.random.default_rng(3407)

        def build_prefix(cands, num_shots):
            if len(cands) <= num_shots:
                chosen = cands
            else:
                idx = rng.choice(len(cands), size=num_shots, replace=False)
                chosen = [cands[i] for i in idx]
            # 用换行拼起来就行，反正 tokenizer 会再切
            return "\n\n".join(chosen)

        # 非成员前缀：每个 mode 只建一次
        if training_mode not in _RECALL_PREFIX_NONMEM:
            print(
                f"[cascade] building prefixes for ReCaLL / Con-ReCaLL, "
                f"dataname={dataname}, mode={training_mode}"
            )
            _RECALL_PREFIX_NONMEM[training_mode] = build_prefix(nonmem_pool, RECALL_NUM_SHOTS)

        # 成员前缀：
        #   - con_recall：每次调用都可以重建（和旧逻辑一样）
        #   - recall：如果当前 mode 还没有，就先建一份，方便之后 con_recall 复用
        if method == "con_recall" or training_mode not in _RECALL_PREFIX_MEM:
            _RECALL_PREFIX_MEM[training_mode] = build_prefix(mem_pool, RECALL_NUM_SHOTS)

        prefix_nm = _RECALL_PREFIX_NONMEM[training_mode]
        prefix_mem = _RECALL_PREFIX_MEM[training_mode]

        print(
            "[cascade] MI method =",
            "ReCaLL" if method == "recall" else "Con-ReCaLL"
        )
        print(
            f"[cascade] ReCaLL shots = {RECALL_NUM_SHOTS}, "
            f"gamma={CONRECALL_GAMMA:.3f}"
        )

        # 计算带非成员前缀的 NLL
        loss_nm = conditional_sentence_loss(
            M1,
            tokenizer,
            texts,
            prefix=prefix_nm,
            max_length=max_length,
            batch_size=8,
        )

        if method == "recall":
            # ReCaLL: score = loss_cond_nonmember / loss_uncond
            mi_scores = loss_nm / loss_M1
        else:
            # Con-ReCaLL 还需要成员前缀条件 loss
            loss_mem = conditional_sentence_loss(
                M1,
                tokenizer,
                texts,
                prefix=prefix_mem,
                max_length=max_length,
                batch_size=8,
            )
            # s(x,M) = (loss_nm - gamma * loss_mem) / loss_un
            mi_scores = (loss_nm - CONRECALL_GAMMA * loss_mem) / loss_M1

    else:
        raise ValueError(
            f"[cascade] 未知 CASCADE_MI_ATTACK={MI_METHOD}，"
            f"只支持 loss / zlib / ref / min-k++ / recall / con-recall"
        )

    # debug: 看一下分布
    print(
        f"[debug] MI stats:  min={mi_scores.min():.4f}, "
        f"max={mi_scores.max():.4f}, mean={mi_scores.mean():.4f}"
    )
    print(
        f"[debug] MIU stats: min={miu_scores.min():.4f}, "
        f"max={miu_scores.max():.4f}, mean={miu_scores.mean():.4f}"
    )

    return mi_scores, miu_scores, labels_class


def fit_thresholds_from_shadow(mi_shadow, miu_shadow, labels_shadow):
    """
    shadow 集上拟合两个阈值 x, y：
      bit1: pretrain membership => Y,B=1; Z,N=0，用 MI 做二分类
      bit2: update  membership => Y,Z=1; B,N=0，用 MIU 做二分类

    这里约定：MI / MIU score 越大越像“member”，所以 greater_is_one=True。
    """
    labels_shadow = np.asarray(labels_shadow, dtype=np.int32)

    # bit1: pretrain membership => Y,B=1; Z,N=0
    b1_true = np.isin(labels_shadow, [0, 1]).astype(int)
    # bit2: update membership  => Y,Z=1; B,N=0
    b2_true = np.isin(labels_shadow, [0, 2]).astype(int)

    # 1) 用 MI 区分 bit1
    x, fpr_x, tpr_x = choose_threshold(mi_shadow, b1_true, greater_is_one=True)
    print(f"[cascade] x (MI for bit1)  = {x:.4f}, FPR={fpr_x:.4f}, TPR={tpr_x:.4f}")

    # 2) 用 MIU 区分 bit2（全局阈值）
    y, fpr_y, tpr_y = choose_threshold(miu_shadow, b2_true, greater_is_one=True)
    print(f"[cascade] y (MIU for bit2) = {y:.4f}, FPR={fpr_y:.4f}, TPR={tpr_y:.4f}")

    return x, y


def eval_on_target(mi_t, miu_t, labels_t, x, y):
    labels_t = np.asarray(labels_t, dtype=np.int32)

    # ===== 真 bit =====
    # 类 0,1: pretrain_bit = 1；类 2,3: pretrain_bit = 0
    # 类 0,2: finetune_bit = 1；类 1,3: finetune_bit = 0
    b1_true = np.isin(labels_t, [0, 1]).astype(int)  # [N]
    b2_true = np.isin(labels_t, [0, 2]).astype(int)  # [N]

    # ===== 预测 bit =====
    # bit1：在 MI 空间里，用 (mi >= x) 判 pretrain member
    b1_pred = (mi_t >= x).astype(int)

    # bit2：在 MIU 空间里，用 (miu >= y) 判 update member（不再分支）
    b2_pred = (miu_t >= y).astype(int)

    # ===== bit1 / bit2 ACC =====
    bit1_acc = (b1_pred == b1_true).mean()
    bit2_acc = (b2_pred == b2_true).mean()
    print(f"[cascade] bit1 (pretrain) acc = {bit1_acc:.4f}")
    print(f"[cascade] bit2 (update)   acc = {bit2_acc:.4f}")

    # ===== 4 类预测 y_pred =====
    # (1,1)->0(Y), (1,0)->1(B), (0,1)->2(Z), (0,0)->3(N)
    y_pred = np.zeros_like(labels_t)
    y_pred[(b1_pred == 1) & (b2_pred == 1)] = 0
    y_pred[(b1_pred == 1) & (b2_pred == 0)] = 1
    y_pred[(b1_pred == 0) & (b2_pred == 1)] = 2
    y_pred[(b1_pred == 0) & (b2_pred == 0)] = 3

    # 4x4 confusion（看个分布，爱看就保留）
    cm4 = confusion_matrix(labels_t, y_pred, labels=[0, 1, 2, 3])
    print("\n[cascade] 4x4 confusion (rows=true class, cols=pred class):")
    print(cm4)

    # ===== ACC: 4-class 整体准确率 =====
    acc4 = (y_pred == labels_t).mean()
    print(f"[cascade] ACC(4-class)       = {acc4:.4f}")

    # ===== macro F1: 4 类宏平均 F1 =====
    try:
        f1_macro = f1_score(labels_t, y_pred, labels=[0, 1, 2, 3], average="macro")
    except ValueError:
        f1_macro = float("nan")
    print(f"[cascade] macro F1(4-class)  = {f1_macro:.4f}")

    # ===== macro AUC: 两个 bit 的 AUC 平均 =====
    # bit1: pretrain member vs non-member，score = mi_t
    # bit2: update  member vs non-member，score = miu_t
    try:
        auc_b1 = roc_auc_score(b1_true, mi_t)
        auc_b2 = roc_auc_score(b2_true, miu_t)
        auc_macro = 0.5 * (auc_b1 + auc_b2)
    except ValueError:
        # 某个 bit 只有一个类的话 roc_auc_score 会炸，这里给个 nan
        auc_macro = float("nan")

    print(f"[cascade] macro AUC(bits)    = {auc_macro:.4f}")


def main():
    """
    用法：
      CASCADE_MI_ATTACK=loss       python cascade_mia.py <model_key> <dataname> [<shadow_model_key>]
      CASCADE_MI_ATTACK=zlib       python cascade_mia.py <model_key> <dataname> [<shadow_model_key>]
      CASCADE_MI_ATTACK=ref        python cascade_mia.py <model_key> <dataname> [<shadow_model_key>]
      CASCADE_MI_ATTACK="min-k++"  python cascade_mia.py <model_key> <dataname> [<shadow_model_key>]
      CASCADE_MI_ATTACK=recall     python cascade_mia.py <model_key> <dataname> [<shadow_model_key>]
      CASCADE_MI_ATTACK=con-recall python cascade_mia.py <model_key> <dataname> [<shadow_model_key>]

    其中：
      - model_key: target 模型（util.model_dict 的 key，例如 pythia / llama / gpt-neo / olmo）
      - shadow_model_key: shadow 模型（若省略，则默认与 model_key 相同）
      - dataname: agnews / onion / oloma / mimir / wikimia（主代码里支持的）
    """
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <model_key> <dataname> [<shadow_model_key>]")
        sys.exit(1)

    model_key = sys.argv[1]
    dataname = sys.argv[2]
    shadow_model_key = sys.argv[3] if len(sys.argv) > 3 else model_key

    print(
        f"[cascade] target_model={model_key}, shadow_model={shadow_model_key}, "
        f"dataname={dataname}, mi_method={MI_METHOD}"
    )

    # 1) shadow：算 MI/MIU 并拟合 x,y
    print("\n==== [cascade] Shadow: compute scores & fit thresholds ====")
    mi_s, miu_s, lab_s = compute_scores_for_mode(
        model_key,
        dataname,
        "shadow",
        shadow_model_key=shadow_model_key,
    )
    x, y = fit_thresholds_from_shadow(mi_s, miu_s, lab_s)
    print(f"[cascade] final thresholds: x={x:.4f}, y={y:.4f}")

    # 2) target：算 MI/MIU 并评估
    print("\n==== [cascade] Target: compute scores & evaluate ====")
    mi_t, miu_t, lab_t = compute_scores_for_mode(
        model_key,
        dataname,
        "target",
        shadow_model_key=shadow_model_key,
    )
    eval_on_target(mi_t, miu_t, lab_t, x, y)


if __name__ == "__main__":
    main()
