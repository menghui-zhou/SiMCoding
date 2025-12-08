#!/bin/bash

# Define configs
lcr_list=(0)
eps_list=(0.5 53.28)

for lcr in "${lcr_list[@]}"; do
    if [[ "$lcr" == "0" ]]; then
        save_dir='./saved_models/ImageNette/Label_Noise_0/'
    else
        save_dir='./saved_models/ImageNette/Label_Noise_0.2/'
    fi



    # RL experiments
    for eps in "${eps_list[@]}"; do
        echo "Running RL experiment with lcr=$lcr, eps=$eps"
        python train_sup_RL.py \
            --eps $eps \
            --epo 500 \
            --lcr $lcr \
            --lr 0.01 \
            --fd 512 \
            --bs 400 \
            --trail '0' \
            --save_fea_ana_freq 500 \
            --arch 'resnet18' \
            --data 'imagenette' \
            --print_freq 23 \
            --transform 'imagenet_train' \
            --test_transform 'imagenet_test' \
            --save_dir $save_dir
    done
done

echo "All experiments on Imagenette complete."
