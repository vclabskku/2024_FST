EXP_NAME=dinov2_large
GPU=0

python train.py \
    --model_name dinov2_vitl14 \
    --train_list train \
    --val_list val \
    --loss_function CrossEntropyLoss \
    --image_cut crop \
    --resolution 448 \
    --gpu ${GPU} \
    --epochs 50 \
    --resume_weights ./ckpts/dinov2_large/dinov2_large_epoch49.pth \
    --eval \
    --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log