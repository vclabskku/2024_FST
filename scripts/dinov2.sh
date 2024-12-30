EXP_NAME=dinov2_base
GPU=0

python train.py \
    --model_name dinov2_vits14 \
    --train_list train \
    --val_list val \
    --loss_function CrossEntropyLoss \
    --image_cut crop \
    --resolution 224 \
    --gpu ${GPU} \
    --epochs 100 \
    --lr_list 0.1 \
    --last_blocks_list 1 \
    --avg_pool_list False \
    --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log