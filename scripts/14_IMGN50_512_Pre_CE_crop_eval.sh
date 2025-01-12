EXP_NAME=14_IMGN50_512_Pre_CE_crop
GPU=0

python train.py \
    --model_name resnet \
    --train_list train \
    --val_list test \
    --test_list test \
    --depth 50 \
    --resume_weights /mnt/hdd0/FST/2024_FST/ckpts/14_IMGN50_512_Pre_CE_crop/14_IMGN50_512_Pre_CE_crop_epoch3.pth \
    --eval \
    --loss_function CrossEntropyLoss \
    --image_cut crop \
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