EXP_NAME=19_IMGN50_512_Pre_BS_resize_lorote0.1
GPU=1

python train.py \
    --model_name resnet \
    --train_list train \
    --val_list val \
    --depth 50 \
    --resume_weights /mnt/hdd0/FST/_prev_20220726/weights/imagenet/resnet50-11ad3fa6.pth \
    --loss_function LogitAdjust \
    --image_cut resize \
    --pretext lorotE \
    --pretext_ratio 0.1 \
    --gpu ${GPU} \
    --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log

# model_name
## cifar imagenet CNN

# loss_function
## CrossEntropyLoss LogitAdjust FocalLoss

# image_cut
## crop resize

# Pretext task
## LorotE LorotI