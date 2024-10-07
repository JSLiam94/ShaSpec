#!/bin/sh

# 获取输入参数，即 GPU 设备编号
gpu_device=$1

# 原始脚本中定义的其他变量
time=$(date "+%Y%m%d-%H%M%S")
name="Eval_mode1_BraTS24_ShaSpecval"

# 函数：执行单次评估
run_evaluation() {
    local mode="$1"
    CUDA_VISIBLE_DEVICES=${gpu_device} python eval.py \
        --input_size=80,160,160 \
        --num_classes=3 \
        --data_list=val.csv \
        --weight_std=True \
        --restore_from=snapshots/BraTS24_ShaSpecval/best.pth \
        --mode="${mode}" > logs/${time}_train_${name}_${mode}.log
}

# 手动列出所有可能的模式组合
modes="
0,1,2,3
0
1
2
3
0,1
0,2
0,3
1,2
1,3
2,3
0,1,2
0,1,3
0,2,3
1,2,3
"

# 遍历所有模式组合并执行评估
for mode in $modes; do
    run_evaluation "$mode"
done
