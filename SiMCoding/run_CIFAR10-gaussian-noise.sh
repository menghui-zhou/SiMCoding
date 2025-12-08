#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_cifar10.sh [lcr]
lcr="${1:-0.0}"

# Define configs
feature_noise_list=(0.04 0.08 0.12 0.16 0.20)
eps_list=(0.5 53.28)

for fnoise in "${feature_noise_list[@]}"; do
    save_dir="./saved_models/CIFAR10/Feature_Noise_${fnoise}/"
    mkdir -p "$save_dir"

    # RL experiments
    for eps in "${eps_list[@]}"; do
        echo "Running RL experiment with lcr=${lcr}, eps=${eps}, noise_std=${fnoise}"
        python train_sup_RL.py \
            --eps "$eps" \
            --epo 500 \
            --lcr "$lcr" \
            --lr 0.01 \
            --fd 512 \
            --trail '0' \
            --noise_std "$fnoise" \
            --save_fea_ana_freq 500 \
            --arch 'resnet18' \
            --data 'cifar10' \
            --print_freq 50 \
            --transform 'default' \
            --save_dir "$save_dir"
    done
done

echo "All experiments on CIFAR-10 complete."
