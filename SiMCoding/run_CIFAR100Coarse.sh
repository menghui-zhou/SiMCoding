

lcr_list=(0.1 0.2 0.3 0.4 0.5)
eps_list=(0.5 38.16)

for lcr in "${lcr_list[@]}"; do

    save_dir="./saved_models/CIFAR20/Label_Noise_${lcr}/"


    # RL experiments
    for eps in "${eps_list[@]}"; do
        echo "Running RL experiment with lcr=$lcr, eps=$eps"
        python train_sup_RL.py \
            --eps $eps \
            --epo 500 \
            --lcr $lcr \
            --lr 0.01 \
            --fd 512 \
            --trail '0' \
            --save_fea_ana_freq 500 \
            --arch 'resnet18' \
            --data 'cifar100coarse' \
            --print_freq 50 \
            --transform 'default' \
            --save_dir $save_dir
    done
done

echo "All experiments on CIFAR-20 complete."

