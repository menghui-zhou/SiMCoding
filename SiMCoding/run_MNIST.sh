#!/bin/bash

# for MNIST experiments with dynamic eps calculation

# Values for lcr
for lcr in 0 ; do

    eps_list=(0.5   6.4)

    for eps in "${eps_list[@]}"; do
        echo "Running RL experiment with lcr=$lcr, eps=$eps"
        python train_sup_RL.py \
            --rough_alpha 0 \
            --eps $eps \
            --epo 500 \
            --lcr $lcr \
            --lr 0.01 \
            --fd 64 \
            --trail '0' \
            --save_fea_ana_freq 500 \
            --arch 'mnistnet' \
            --data 'mnist' \
            --save_dir './saved_models/MNIST/Label_Noise_0' \
            --print_freq 50 \
            --transform 'test'
    done
done

echo "All experiments on MNIST complete."
