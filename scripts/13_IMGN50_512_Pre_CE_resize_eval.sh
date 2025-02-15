EXP_NAME=13_IMGN50_512_Pre_CE_resize
GPU=0

python train.py \
    --model_name resnet \
    --train_list train \
    --val_list test \
    --test_list test \
    --depth 50 \
    --resume_weights /mnt/hdd0/FST/2024_FST/ckpts/13_IMGN50_512_Pre_CE_resize/13_IMGN50_512_Pre_CE_resize_epoch37.pth   \
    --eval \
    --loss_function CrossEntropyLoss \
    --image_cut resize \
    --gpu ${GPU} \
    --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log

# model_name
## cifar imagenet CNN

# loss_function
## CrossEntropyLoss LogitAdjust FocalLoss

# image_cut
## crop resize

# Pretext task
## lorotE lorotI