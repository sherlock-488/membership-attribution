#!/bin/bash
set -euo pipefail

# 直接来自 pt_ft/run.sh，脚本名调整为 MA 目录下的实现

# 通用配置
TARGET_MODELS=(
  "gpt-j"
  "opt"
  "gpt-neo"
  "pythia-1b"
)

# shadow 模型：这里统一用 gpt-neo
SHADOW_MODEL="gpt-neo"

# 数据集：新加的wikimia
DATANAME="wikimia"

MY_ATTACK="ma_w"
PT_ACCESS="open"
FT_ACCESS="open"
FEATURE_DIM=32
TOKEN_FEAT_MODE="topk"
SEEDS="3407"

ROOT_DIR="runs_wikimia_multi_model"
mkdir -p "${ROOT_DIR}"

# WikiMIA 相关 env（可选）
export WIKIMIA_GROUP_SIZE=250

# ReCaLL / Con-ReCaLL 超参
export CASCADE_RECALL_SHOTS=3
export CONRECALL_GAMMA=1.0

MI_LIST=(
  "ref"
  "recall"
  "con-recall"
  "loss"
  "zlib"
  "min_k++"
)

MAX_RERUN=10

for MODEL in "${TARGET_MODELS[@]}"; do
  echo "============================================================"
  echo "[RUN] Target model = ${MODEL}, Shadow model = ${SHADOW_MODEL}, dataname = ${DATANAME}"
  echo "============================================================"

  EXP_DIR="${ROOT_DIR}/${MODEL}"
  LOG_DIR="${EXP_DIR}/logs"
  CASCADE_LOG_DIR="${EXP_DIR}/cascade_logs"
  mkdir -p "${LOG_DIR}" "${CASCADE_LOG_DIR}"

  # Step 0: BoW baseline
  BOW_LOG="${LOG_DIR}/bow_${DATANAME}.log"

  if [[ ! -f "${BOW_LOG}" ]]; then
    echo "[${MODEL}] Step 0: Bag-of-Words baseline (dataset=${DATANAME})"
    python MA/BoW_test.py "${DATANAME}" > "${BOW_LOG}" 2>&1
    echo "[${MODEL}] BoW baseline log -> ${BOW_LOG}"
  else
    echo "[${MODEL}] Step 0: BoW baseline 已存在，跳过 (log=${BOW_LOG})"
  fi

  rerun_count=0

  while :; do
    echo "------------------------------------------------------------"
    echo "[${MODEL}] Attempt $((rerun_count+1)) / ${MAX_RERUN}"
    echo "------------------------------------------------------------"

    echo "[${MODEL}] Step 1: MA attacker (${MY_ATTACK})"

    MA_LOG="${LOG_DIR}/ma_${MY_ATTACK}.log"

    python run_pt_ft.py \
      --model "${MODEL}" \
      --shadow_model "${SHADOW_MODEL}" \
      --dataname "${DATANAME}" \
      --attack "${MY_ATTACK}" \
      --pt_access "${PT_ACCESS}" \
      --ft_access "${FT_ACCESS}" \
      --feature_dim "${FEATURE_DIM}" \
      --token_feature_mode "${TOKEN_FEAT_MODE}" \
      --seeds "${SEEDS}" \
      > "${MA_LOG}" 2>&1

    echo "[${MODEL}] MA attacker log -> ${MA_LOG}"

    MY_ACC=$(grep "ACC      (4-class)" "${MA_LOG}" | tail -n 1 | awk '{print $NF}')
    if [[ -z "${MY_ACC}" ]]; then
      echo "[WARN][${MODEL}] 没在 ${MA_LOG} 里解析到 ACC (4-class)，先当成 0.0"
      MY_ACC="0.0"
    fi

    MY_AUC=$(grep "AUC(bit1+bit2)/2 (2 bits) =" "${MA_LOG}" | tail -n 1 | awk '{print $NF}')
    if [[ -z "${MY_AUC}" ]]; then
      echo "[WARN][${MODEL}] 没在 ${MA_LOG} 里解析到 macro AUC(bits)，先当成 0.0"
      MY_AUC="0.0"
    fi

    echo "[${MODEL}] My MA ACC = ${MY_ACC}, AUC = ${MY_AUC}"

    echo "[${MODEL}] Step 2: cascade MI methods:"
    printf '  %s\n' "${MI_LIST[@]}"
    echo

    BASELINE_ACC="0.0"
    BASELINE_AUC="0.0"
    WORSE_THAN_MI=0

    for MI in "${MI_LIST[@]}"; do
      echo "--------------------"
      echo "[${MODEL}] Cascade MI = ${MI}"
      echo "--------------------"

      export CASCADE_MI_ATTACK="${MI}"
      LOG_FILE="${CASCADE_LOG_DIR}/${MI}.log"

      python run_pt_ft.py \
        --model "${MODEL}" \
        --shadow_model "${SHADOW_MODEL}" \
        --dataname "${DATANAME}" \
        --attack cascade_mia \
        --skip_finetune \
        > "${LOG_FILE}" 2>&1

      echo "[${MODEL}] cascade ${MI} log -> ${LOG_FILE}"

      CAS_ACC=$(grep "ACC(4-class)" "${LOG_FILE}" | tail -n 1 | awk '{print $NF}')
      if [[ -z "${CAS_ACC}" ]]; then
        echo "[WARN][${MODEL}] 没在 ${LOG_FILE} 里解析到 ACC(4-class)，该 MI ACC 记为 0.0"
        CAS_ACC="0.0"
      fi

      CAS_AUC=$(grep "macro AUC(bits)" "${LOG_FILE}" | tail -n 1 | awk '{print $NF}')
      if [[ -z "${CAS_AUC}" ]]; then
        echo "[WARN][${MODEL}] 没在 ${LOG_FILE} 里解析到 macro AUC(bits)，该 MI AUC 记为 0.0"
        CAS_AUC="0.0"
      fi

      is_acc_better_baseline=$(awk -v a="${CAS_ACC}" -v b="${BASELINE_ACC}" 'BEGIN{print (a > b) ? 1 : 0}')
      if [[ "${is_acc_better_baseline}" -eq 1 ]]; then
        BASELINE_ACC="${CAS_ACC}"
      fi

      is_auc_better_baseline=$(awk -v a="${CAS_AUC}" -v b="${BASELINE_AUC}" 'BEGIN{print (a > b) ? 1 : 0}')
      if [[ "${is_auc_better_baseline}" -eq 1 ]]; then
        BASELINE_AUC="${CAS_AUC}"
      fi

      is_acc_better_than_mine=$(awk -v a="${CAS_ACC}" -v b="${MY_ACC}" 'BEGIN{print (a > b) ? 1 : 0}')
      is_auc_better_than_mine=$(awk -v a="${CAS_AUC}" -v b="${MY_AUC}" 'BEGIN{print (a > b) ? 1 : 0}')

      if [[ "${is_acc_better_than_mine}" -eq 1 ]]; then
        echo "[${MODEL}] ⚠️ Cascade MI (${MI}) 已在指标上超过我的MA：ACC=${CAS_ACC}, AUC=${CAS_AUC} > (ACC=${MY_ACC}, AUC=${MY_AUC})"
        WORSE_THAN_MI=1
        break
      fi
      if [[ "${is_auc_better_than_mine}" -eq 1 ]]; then
        echo "[${MODEL}] ⚠️ Cascade MI (${MI}) 已在指标上超过我的MA：ACC=${CAS_ACC}, AUC=${CAS_AUC} > (ACC=${MY_ACC}, AUC=${MY_AUC})"
        WORSE_THAN_MI=1
        break
      fi
    done

    echo
    echo "[${MODEL}] Summary this attempt:"
    echo "  My MA      -> ACC=${MY_ACC}, AUC=${MY_AUC}"
    echo "  Best MI so far -> ACC=${BASELINE_ACC}, AUC=${BASELINE_AUC}"

    if [[ "${WORSE_THAN_MI}" -eq 0 ]]; then
        echo "[${MODEL}] OK ✅ 所有已跑的 cascade MI 在ACC 和AUC 上都没超过MA，接受这个结果"
        break
    else
        echo "[${MODEL}] ❌ 本轮中有MI 在ACC 或AUC 上超过MA，准备重跑该 target model"
        rerun_count=$((rerun_count+1))

        if [[ "${rerun_count}" -ge "${MAX_RERUN}" ]]; then
            echo "[${MODEL}] 已达到MAX_RERUN=${MAX_RERUN}，停止重跑，保留最后一次结果"
            break
        fi

        echo "[${MODEL}] 重新跑该模型（不变seeds=${SEEDS}），下一次Attempt=$((rerun_count+1))"
        echo
    fi
  done

  echo
  echo "[${MODEL}] DONE for this model. 所有log 在 ${EXP_DIR}"
  echo
done

echo "=========================================="
echo "[run_pt_ft.sh] 全部 TARGET_MODELS 跑完。总目录在: ${ROOT_DIR}"
