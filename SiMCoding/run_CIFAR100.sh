#!/bin/bash

# Define configs
lcr_list=(0)
eps_list=(0.5 17.07)

fea_dim=512

for lcr in "${lcr_list[@]}"; do
    # 根据 lcr 选择保存目录
    if [[ "$lcr" == "0" ]]; then
        save_dir='./saved_models/CIFAR100x5/Label_Noise_0/'
    else
        save_dir='./saved_models/CIFAR100x5/Label_Noise_0.2/'
    fi



     # RL experiments
    for eps in "${eps_list[@]}"; do
        echo "Running RL experiment with lcr=$lcr, eps=$eps"
        python train_sup_RL.py \
            --eps $eps \
            --epo 500 \
            --lcr $lcr \
            --lr 0.1 \
            --fd $fea_dim \
            --trail '0' \
            --save_fea_ana_freq 500 \
            --arch 'resnet18' \
            --data 'cifar100x5' \
            --print_freq 50 \
            --transform default \
            --save_dir $save_dir
    done

done

echo "All experiments on CIFAR-100 complete."


