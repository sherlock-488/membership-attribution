# util.py (pt_ft pipeline, adapted to MA path)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from torch.nn.functional import softmax
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

# 保存到当前项目下，避免依赖外部绝对路径
SAVE_PATH = './'

model_dict = {
    'gpt-neo': 'EleutherAI/gpt-neo-125m',
    'gpt-j': 'EleutherAI/gpt-j-6B',
    'opt': 'facebook/opt-350m',

    'pythia-1b':   'EleutherAI/pythia-1b-deduped',
    'pythia-1.4b': 'EleutherAI/pythia-1.4b-deduped',
    'pythia-6.9b': 'EleutherAI/pythia-6.9b-deduped',
    'pythia-12b':  'EleutherAI/pythia-12b-deduped',

    'pythia': 'EleutherAI/pythia-1.4b-deduped',

    'llama3': 'meta-llama/Llama-3.2-1B',
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    
    'olmo': 'allenai/OLMo-1B-hf',
}

# ===== OLOMA 数据源与规模 =====
# X：Dolma 里的 Reddit 子集（allenai/dolma, config=v1_6-sample）
OLOMA_X_DATASET = os.getenv('OLOMA_X_DATASET', 'allenai/dolma')
OLOMA_X_CONFIG  = os.getenv('OLOMA_X_CONFIG',  'v1_6-sample')

# Y：Paloma 里的 Dolma 100 Subreddits（allenai/paloma, config=dolma_100_subreddits）
OLOMA_W_DATASET = os.getenv('OLOMA_W_DATASET', 'allenai/paloma')
OLOMA_W_CONFIG  = os.getenv('OLOMA_W_CONFIG',  'dolma_100_subreddits')
OLOMA_W_SPLIT   = os.getenv('OLOMA_W_SPLIT',   'val+test')

# 每类样本数（k）
OLOMA_GROUP_SIZE = int(os.getenv('OLOMA_GROUP_SIZE', '1000'))

# ===== MIMIR：基于The Pile 的pretrain MI benchmark =====
MIMIR_DATASET = os.getenv('MIMIR_DATASET', 'iamgroot42/mimir')
# 单config 默认用full_pile（通过 MIMIR_CONFIG_LIST 控制）
MIMIR_CONFIG  = os.getenv('MIMIR_CONFIG',  'wikipedia_(en)')
# 默认优先尝试 ngram_7_0.2，不行再回退到none（在 _load_mimir_sources 里）
MIMIR_SPLIT   = os.getenv('MIMIR_SPLIT',   'none')

# 全局抽样 config 列表
MIMIR_CONFIG_LIST = os.getenv(
    'MIMIR_CONFIG_LIST',
    'full_pile'
)

# 每类样本数k：默认500
MIMIR_GROUP_SIZE = int(os.getenv('MIMIR_GROUP_SIZE', '500'))

# ===== WikiMIA：基于Wikipedia 的pretrain MI benchmark =====
# HF 数据集：swj0419/WikiMIA，schema: {'input', 'label'}
#   label: 0 = non-member, 1 = member
WIKIMIA_DATASET = os.getenv('WIKIMIA_DATASET', 'swj0419/WikiMIA')
# 文本长度（32/64/128/256 之一），对应 config 名的后缀
WIKIMIA_LENGTH  = int(os.getenv('WIKIMIA_LENGTH', '64'))
# 每类样本数上限
WIKIMIA_GROUP_SIZE = int(os.getenv('WIKIMIA_GROUP_SIZE', '200'))


def _load_oloma_sources():
    """
    X = Dolma v1_6-sample 里的 Reddit 子集（allenai/dolma, config=v1_6-sample）
    Y = Paloma 中Dolma 100 Subreddits 子语料（allenai/paloma, config=dolma_100_subreddits，val+test）

    返回：
      x: 打乱后的 X，只保留 'text'
      w: 打乱后的 Y，只保留 'text'
    """
    # ---- X: Dolma Reddit 子集 ----
    if OLOMA_X_CONFIG is None:
        x = load_dataset(
            OLOMA_X_DATASET,
            split='train',
            trust_remote_code=True,
        )
    else:
        x = load_dataset(
            OLOMA_X_DATASET,
            OLOMA_X_CONFIG,
            split='train',
            trust_remote_code=True,
        )

    if 'source' not in x.column_names:
        raise ValueError(
            f"OLOMA_X={OLOMA_X_DATASET} (config={OLOMA_X_CONFIG}) 没有 'source' 字段，"
            "无法持Reddit 子集筛选。请改成 allenai/dolma v1_6-sample 或别的带 source 的Dolma 版本，"
            "或者在外部预处理好只含 reddit 的数据集再传进来。"
        )

    # 只保留Dolma 里的 Reddit 子集
    x = x.filter(lambda ex: ex['source'] == 'reddit')

    if 'text' not in x.column_names:
        raise ValueError("OLOMA_X 数据集必须包含'text' 列")

    # 全量 reddit 子集乱序，然后只保留 text
    x = x.shuffle(seed=2025).select_columns('text')

    if len(x) < 4 * OLOMA_GROUP_SIZE:
        raise ValueError(
            f"OLOMA_X Reddit 子集样本太少：需要至少{4 * OLOMA_GROUP_SIZE} 条，实际 {len(x)} 条"
        )

    # ---- Y: Paloma Dolma 100 Subreddits ----
    splits = [s for s in OLOMA_W_SPLIT.split('+') if s]
    w_parts = [
        load_dataset(OLOMA_W_DATASET, OLOMA_W_CONFIG, split=split)
        for split in splits
    ]
    w = concatenate_datasets(w_parts)

    if 'text' not in w.column_names:
        raise ValueError("OLOMA_W(Paloma) 数据集必须包含'text' 列")

    # 全量 val+test 乱序，然后只保留 text
    w = w.shuffle(seed=3407).select_columns('text')

    if len(w) < 4 * OLOMA_GROUP_SIZE:
        raise ValueError(
            f"OLOMA_W(top-100 subreddits) 样本太少：需要至少{4 * OLOMA_GROUP_SIZE} 条，实际 {len(w)} 条"
        )

    return x, w


def _oloma_YBZN(x, w, mode: str, group_size: int = OLOMA_GROUP_SIZE):
    """
    OLOMA 的A/B/C/D 切分。
    """
    g = group_size
    if mode == 'target':
        Y = x.select(range(0, g))        # A : <1,1>
        B = x.select(range(g, 2 * g))    # B : <1,0>
        Z = w.select(range(0, g))        # C : <0,1>
        N = w.select(range(g, 2 * g))    # D : <0,0>
    elif mode == 'shadow':
        Y = x.select(range(2 * g, 3 * g))  # A'
        B = x.select(range(3 * g, 4 * g))  # B'
        Z = w.select(range(2 * g, 3 * g))  # C'
        N = w.select(range(3 * g, 4 * g))  # D'
    else:
        raise ValueError("Invalid training mode (must be 'target' or 'shadow')")
    return Y, B, Z, N


def _load_mimir_sources():
    """
    抽取 MIMIR 的member / nonmember：
      - 从MIMIR_CONFIG_LIST 里的 config 读数据
      - 对每个config 优先尝试使用 MIMIR_SPLIT（例如ngram_7_0.2），不行再回退到'none'
      - 把所有member 拼成一个大集合 X
      - 把所有nonmember 拼成一个大集合 W
      - 这里的shuffle 用固定seed，这样target / shadow 看到的是同一套顺序
    """
    config_names = [
        c.strip() for c in MIMIR_CONFIG_LIST.split(',')
        if c.strip()
    ]

    x_parts = []
    w_parts = []
    for cfg in config_names:
        ds = None

        for split_name in [MIMIR_SPLIT, 'none']:
            try:
                ds = load_dataset(
                    MIMIR_DATASET,
                    cfg,
                    split=split_name,
                    trust_remote_code=True,
                )
                break
            except ValueError:
                continue

        if ds is None:
            continue

        if 'member' not in ds.column_names or 'nonmember' not in ds.column_names:
            continue

        # 当前 config 的member ←text
        x_cfg = ds.remove_columns(
            [c for c in ds.column_names if c != 'member']
        ).rename_column('member', 'text')

        # 当前 config 的nonmember ←text
        w_cfg = ds.remove_columns(
            [c for c in ds.column_names if c != 'nonmember']
        ).rename_column('nonmember', 'text')

        x_parts.append(x_cfg)
        w_parts.append(w_cfg)

    if not x_parts or not w_parts:
        raise ValueError(
            f"MIMIR: 在configs={config_names} 中没找到合法的member/nonmember 数据，"
            "请检查MIMIR_CONFIG_LIST / MIMIR_SPLIT 设置，或者是否已经开通gated dataset 权限。"
        )

    x = concatenate_datasets(x_parts).shuffle(seed=2025)
    w = concatenate_datasets(w_parts).shuffle(seed=3407)

    return x, w


def _mimir_YBZN(x, w, mode: str, group_size: int = MIMIR_GROUP_SIZE):
    """
    通用的Y/B/Z/N 切分工具（原本为 MIMIR，这里也被WikiMIA 复用），要求：
      - target 和shadow 使用 **不重叠** 的样本
      - 两边各自都有 Y,B,Z,N 四类，大小相同

    具体做法：
      - 在调用方保证 x / w 各自 shuffle 过
      - 这里再根据长度动态确定g：

          g <= group_size
          g <= len(x) // 4
          g <= len(w) // 4

        然后把x、w 各自切成 4 段，每段长度 g：

          target:
            x[0:g]      -> Y_t
            x[g:2g]     -> B_t
            w[0:g]      -> Z_t
            w[g:2g]     -> N_t

          shadow:
            x[2g:3g]    -> Y_s
            x[3g:4g]    -> B_s
            w[2g:3g]    -> Z_s
            w[3g:4g]    -> N_s
    """
    if mode not in ['target', 'shadow']:
        raise ValueError("Invalid training mode (must be 'target' or 'shadow')")

    max_g_x = len(x) // 4
    max_g_w = len(w) // 4
    g = min(group_size, max_g_x, max_g_w)

    if g == 0:
        raise ValueError(
            f"YBZN: 数据太少，无法按 target/shadow 拆成 4 段"
            f"(len(x)={len(x)}, len(w)={len(w)}, group_size={group_size})"
        )

    if mode == 'target':
        Y = x.select(range(0 * g, 1 * g))   # A : <1,1>
        B = x.select(range(1 * g, 2 * g))   # B : <1,0>
        Z = w.select(range(0 * g, 1 * g))   # C : <0,1>
        N = w.select(range(1 * g, 2 * g))   # D : <0,0>
    else:  # mode == 'shadow'
        Y = x.select(range(2 * g, 3 * g))   # A'
        B = x.select(range(3 * g, 4 * g))   # B'
        Z = w.select(range(2 * g, 3 * g))   # C'
        N = w.select(range(3 * g, 4 * g))   # D'

    return Y, B, Z, N


def _load_wikimia_sources():
    """
    加载 WikiMIA：
      HF: swj0419/WikiMIA
        - 'input': 文本
        - 'label': 0 = non-member, 1 = member

    返回：
      x: member 文本，只含'text' 列
      w: non-member 文本，只含'text' 列
    """
    length = WIKIMIA_LENGTH
    split_name = f"WikiMIA_length{length}"

    ds = load_dataset(
        WIKIMIA_DATASET,
        split=split_name,
        trust_remote_code=True,
    )

    if 'input' not in ds.column_names or 'label' not in ds.column_names:
        raise ValueError(
            f"WikiMIA 数据集{WIKIMIA_DATASET} (split={split_name}) 必须包含 'input' 和'label' 列"
        )

    mem = ds.filter(lambda ex: ex['label'] == 1)
    nonmem = ds.filter(lambda ex: ex['label'] == 0)

    if len(mem) < 4 or len(nonmem) < 4:
        raise ValueError(
            f"WikiMIA: member/nonmember 数据太少："
            f"member={len(mem)}, nonmember={len(nonmem)}"
        )

    mem = mem.select_columns(['input']).rename_column('input', 'text').shuffle(seed=2025)
    nonmem = nonmem.select_columns(['input']).rename_column('input', 'text').shuffle(seed=3407)

    return mem, nonmem


def load_train_data(dataname, training_mode, training_stage):
    if dataname == 'agnews':
        dataset = load_dataset('fancyzhx/ag_news', split='train')
        data = dataset.shuffle(seed=2023).select_columns('text')
        if training_mode not in ['target', 'shadow']:
            raise ValueError('Invalid training mode')
        data = data.select(range(10000)) if training_mode == 'target' else data.select(range(10000, 20000))
        D_1 = data.select(range(0, 2500))
        D_2 = data.select(range(2500, 5000))
        D_3 = data.select(range(5000, 7500))
        if training_stage not in ['first', 'second']:
            raise ValueError('Invalid training stage')
        return concatenate_datasets([D_2, D_3]) if training_stage == 'first' else concatenate_datasets([D_1, D_2])

    elif dataname == 'onion':
        dataset = load_dataset('Biddls/Onion_News', split='train')
        data = dataset.shuffle(seed=2023).select_columns('text')
        if training_mode not in ['target', 'shadow']:
            raise ValueError('Invalid training mode')
        data = data.select(range(10000)) if training_mode == 'target' else data.select(range(10000, 20000))
        D_1 = data.select(range(0, 2500))
        D_2 = data.select(range(2500, 5000))
        D_3 = data.select(range(5000, 7500))
        if training_stage not in ['first', 'second']:
            raise ValueError('Invalid training stage')
        return concatenate_datasets([D_2, D_3]) if training_stage == 'first' else concatenate_datasets([D_1, D_2])

    elif dataname == 'oloma':
        # OLOMA：first（基线）不再训练；second 用于 M2 训练：Y ∪ Z
        if training_mode not in ['target', 'shadow']:
            raise ValueError('Invalid training mode')
        if training_stage not in ['first', 'second']:
            raise ValueError('Invalid training stage')
        x, w = _load_oloma_sources()
        Y, B, Z, N = _oloma_YBZN(x, w, mode=training_mode, group_size=OLOMA_GROUP_SIZE)
        if training_stage == 'first':
            return concatenate_datasets([B])  # 不用到，只是保持 API
        else:
            return concatenate_datasets([Y, Z])  # (A ∪ C)

    elif dataname == 'mimir':
        # MIMIR：类比OLOMA，但 target/shadow 使用不重叠的 Y/B/Z/N 段
        if training_mode not in ['target', 'shadow']:
            raise ValueError('Invalid training mode')
        if training_stage not in ['first', 'second']:
            raise ValueError('Invalid training stage')
        x, w = _load_mimir_sources()
        Y, B, Z, N = _mimir_YBZN(x, w, mode=training_mode, group_size=MIMIR_GROUP_SIZE)
        if training_stage == 'first':
            return concatenate_datasets([B])  # 占位，不实际使用
        else:
            return concatenate_datasets([Y, Z])  # (A ∪ C)

    elif dataname == 'wikimia':
        # WikiMIA：完全复用MIMIR 的Y/B/Z/N 逻辑
        if training_mode not in ['target', 'shadow']:
            raise ValueError('Invalid training mode')
        if training_stage not in ['first', 'second']:
            raise ValueError('Invalid training stage')
        x, w = _load_wikimia_sources()
        Y, B, Z, N = _mimir_YBZN(x, w, mode=training_mode, group_size=WIKIMIA_GROUP_SIZE)
        if training_stage == 'first':
            return concatenate_datasets([B])  # 占位，不实际使用
        else:
            return concatenate_datasets([Y, Z])  # (A ∪ C)

    else:
        raise ValueError('Invalid dataset name')


def load_eval_data(dataname, training_mode):
    if dataname == 'agnews':
        dataset = load_dataset('fancyzhx/ag_news', split='train')
        data = dataset.shuffle(seed=2023).select_columns('text')
        return data.select(range(10000)) if training_mode == 'target' else data.select(range(10000, 20000))

    elif dataname == 'onion':
        dataset = load_dataset('Biddls/Onion_News', split='train')
        data = dataset.shuffle(seed=2023).select_columns('text')
        return data.select(range(10000)) if training_mode == 'target' else data.select(range(10000, 20000))

    elif dataname == 'oloma':
        x, w = _load_oloma_sources()
        Y, B, Z, N = _oloma_YBZN(x, w, mode=training_mode, group_size=OLOMA_GROUP_SIZE)
        # 评估严格拼[Y, B, Z, N] = [A, B, C, D]
        return concatenate_datasets([Y, B, Z, N])

    elif dataname == 'mimir':
        x, w = _load_mimir_sources()
        Y, B, Z, N = _mimir_YBZN(x, w, mode=training_mode, group_size=MIMIR_GROUP_SIZE)
        # 同样评估严格拼[A, B, C, D]
        return concatenate_datasets([Y, B, Z, N])

    elif dataname == 'wikimia':
        x, w = _load_wikimia_sources()
        Y, B, Z, N = _mimir_YBZN(x, w, mode=training_mode, group_size=WIKIMIA_GROUP_SIZE)
        # 同样评估严格拼[A, B, C, D]
        return concatenate_datasets([Y, B, Z, N])

    else:
        raise ValueError('Invalid dataset name')


class shadowDataset(Dataset):
    def __init__(self, filepath, baseline=False, num_per_class=2500):
        self.preds = torch.load(filepath)
        self.num_per_class = num_per_class
        if baseline:
            self.preds_baseline = self.preds[:5000] + self.preds[7500:]
            self.preds = self.preds_baseline
        
    def get_dim(self):
        return self.preds[0].shape[1]
    
    def __len__(self):
        return len(self.preds)
    
    def __getitem__(self, idx):
        return self.preds[idx], idx//self.num_per_class


def get_last_prob(model, tokens):
    outputs = model(**tokens)
    last_layer_prob = softmax(outputs.logits[..., :-1, :].float(), -1)
    return last_layer_prob


def _log_value(probs, small_value=1e-30):
    probs = probs.float()
    return -torch.log(torch.clamp(probs, min=small_value))


def contrastive_prob(prob_1, prob_2):
    delta_logits = _log_value(prob_1)-_log_value(prob_2)
    return softmax(delta_logits, -1)


def get_token_masks(prob, label):
    prob = prob.view(label.shape[0], -1)
    prob_true = prob[torch.arange(label.size(0)).unsqueeze(1), label.unsqueeze(1)].view(-1)
    return torch.where(prob_true>0, -1, 1)


def get_cont_token_masks(prob_1, prob_2, label):
    prob_1 = prob_1.view(label.shape[0], -1)
    prob_2 = prob_2.view(label.shape[0], -1)
    prob_1_true = prob_1[torch.arange(label.size(0)).unsqueeze(1), label.unsqueeze(1)].view(-1)
    prob_2_true = prob_2[torch.arange(label.size(0)).unsqueeze(1), label.unsqueeze(1)].view(-1)
    return _log_value(prob_2_true)-_log_value(prob_1_true)


def m_entr_sum(probs, true_labels):
    log_probs = _log_value(probs)
    reverse_probs = 1-probs
    log_reverse_probs = _log_value(1-probs)
    modified_probs = torch.clone(probs)
    modified_probs[torch.arange(true_labels.size(0)).unsqueeze(1), true_labels.unsqueeze(1)] = reverse_probs[torch.arange(true_labels.size(0)).unsqueeze(1), true_labels.unsqueeze(1)]
    modified_log_probs = torch.clone(log_reverse_probs)
    modified_log_probs[torch.arange(true_labels.size(0)).unsqueeze(1), true_labels.unsqueeze(1)] = log_probs[torch.arange(true_labels.size(0)).unsqueeze(1), true_labels.unsqueeze(1)]
    return -torch.sum(torch.mul(modified_probs, modified_log_probs), axis=1)


def entr_sum(probs):
    return -torch.sum(torch.mul(probs, _log_value(probs)), -1)
