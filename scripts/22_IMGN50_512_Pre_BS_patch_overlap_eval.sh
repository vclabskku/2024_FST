EXP_NAME=22_IMGN50_512_Pre_BS_resize_overlap_4
GPU=0

python train.py \
    --model_name ResNet_patch_overlap \
    --train_list train \
    --val_list test \
    --test_list test \
    --depth 50 \
    --resume_weights /mnt/ssd0/WJ/lab_assn/FST/suho/2024_FST/ckpts/22_IMGN50_512_Pre_BS_resize_overlap_4/22_IMGN50_512_Pre_BS_resize_overlap_4_epoch28.pth \
    --loss_function LogitAdjust \
    --image_cut patch \
    --gpu ${GPU} \
    --patch_num 4 \
    --overlap 50 \
    --epochs 100 \
    --eval \
    --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log


# model_name
## cifar imagenet CNN

# loss_function
## CrossEntropyLoss LogitAdjust FocalLoss

# image_cut
## crop resize

# Pretext task
## lorotE lorotI