GPU=0

EXP_NAME=21_IMGN50_512_Pre_BS_resize_16patch
python train.py \
    --model_name ResNet_patch16 \
    --train_list train \
    --val_list test \
    --test_list test \
    --depth 50 \
    --resume_weights /mnt/hdd0/FST/_prev_20220726/weights/imagenet/resnet50-11ad3fa6.pth \
    --loss_function LogitAdjust \
    --image_cut patch \
    --gpu ${GPU} \
    --patch_num 16 \
    --epochs 100 \
    --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log

# model_name
## cifar imagenet CNN

# loss_function
## CrossEntropyLoss LogitAdjust FocalLoss

# image_cut
## crop resize

# Pretext task
## lorotE lorotI