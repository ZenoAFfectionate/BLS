#!/bin/bash
# ============================================================
# BLS & ARBN 实验脚本
# 4 个数据集 x 2 个模型 x 5 个不平衡因子 = 40 个实验
# 用法: bash scripts/run_all_experiments.sh
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${PROJECT_DIR}/data"
LOG_DIR="${PROJECT_DIR}/logs"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

# 实验参数
DATASETS=("MNIST" "FashionMNIST" "CIFAR10" "CIFAR100")
MODELS=("bls" "arbn")
IMB_FACTORS=(1 10 50 100 200)

# 模型参数
FEATURE_TIMES=10
ENHANCE_TIMES=10
REG=0.01
SEED=42

mkdir -p "${LOG_DIR}"

RESULT_FILE="${LOG_DIR}/results_summary.txt"
echo "============================================================" > "${RESULT_FILE}"
echo "BLS & ARBN 实验结果汇总" >> "${RESULT_FILE}"
echo "运行时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "${RESULT_FILE}"
echo "参数: feature_times=${FEATURE_TIMES}, enhance_times=${ENHANCE_TIMES}, reg=${REG}" >> "${RESULT_FILE}"
echo "============================================================" >> "${RESULT_FILE}"
echo "" >> "${RESULT_FILE}"

total=0
passed=0
failed=0
skipped=0

run_one() {
    local dataset="$1" model="$2" imb="$3"
    local log_file="${LOG_DIR}/${dataset}_${model}_IF${imb}.log"
    local exp_name="${dataset}_${model}_IF${imb}"

    # 跳过已成功的实验
    if [ -f "${log_file}" ] && grep -q "Test Accuracy" "${log_file}" 2>/dev/null; then
        echo "  SKIP ${exp_name} (already done)"
        skipped=$((skipped + 1))
        return 0
    fi

    total=$((total + 1))
    echo "  [${total}/40] RUN ${exp_name} ..."

    local feature_size=256

    local cmd="python ${PROJECT_DIR}/main.py \
        --dataset ${dataset} \
        --model ${model} \
        --data_root ${DATA_ROOT} \
        --imbalance_factor ${imb} \
        --feature_size ${feature_size} \
        --feature_times ${FEATURE_TIMES} \
        --enhance_times ${ENHANCE_TIMES} \
        --reg ${REG} \
        --seed ${SEED}"

    if [ "${model}" = "arbn" ]; then
        cmd="${cmd} --class_weight_beta 0.5"
    fi

    # 运行实验
    ${cmd} > "${log_file}" 2>&1
    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        local acc=$(grep "Test Accuracy" "${log_file}" | tail -1 | sed 's/.*Test Accuracy: //' | sed 's/%.*//')
        local recall=$(grep "Recall (macro)" "${log_file}" | tail -1 | sed 's/.*Recall (macro): //' | sed 's/%.*//')
        local f1=$(grep "F1 (macro)" "${log_file}" | tail -1 | sed 's/.*F1 (macro): //' | sed 's/%.*//')
        local top5=$(grep "Top-5 Accuracy" "${log_file}" | tail -1 | sed 's/.*Top-5 Accuracy: //' | sed 's/%.*//')
        [ -z "${acc}" ] && acc="N/A"
        local info_str="acc=${acc}%"
        [ -n "${recall}" ] && info_str="${info_str}, recall(m)=${recall}%"
        [ -n "${f1}" ] && info_str="${info_str}, f1(m)=${f1}%"
        [ -n "${top5}" ] && info_str="${info_str}, top5=${top5}%"
        echo "     PASS (${info_str})"
        passed=$((passed + 1))
    else
        echo "     FAIL (exit code: ${exit_code})"
        failed=$((failed + 1))
    fi
}

echo "============================================"
echo "  BLS & ARBN Experiment Suite"
echo "  Datasets: ${DATASETS[*]}"
echo "  Models: ${MODELS[*]}"
echo "  Imbalance Factors: ${IMB_FACTORS[*]}"
echo "============================================"
echo ""

# 按数据集迭代
for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: ${dataset} ---"
    for imb in "${IMB_FACTORS[@]}"; do
        for model in "${MODELS[@]}"; do
            run_one "${dataset}" "${model}" "${imb}"
        done
    done
    echo ""
done

echo "============================================================"
echo "总计: ${total}  通过: ${passed}  失败: ${failed}  跳过: ${skipped}"
echo "日志目录: ${LOG_DIR}"
echo "============================================================"

# 生成结果表格
python3 "${SCRIPTS_DIR}/collect_results.py" "${LOG_DIR}"

echo ""
echo "结果汇总: ${RESULT_FILE}"
