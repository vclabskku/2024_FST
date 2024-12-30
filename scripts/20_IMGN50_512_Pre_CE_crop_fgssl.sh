EXP_NAME=20_IMGN50_512_Pre_CE_crop_fgssl
GPU=1

python train.py \
    --batch_size 100 \
    --model_name resnet \
    --train_list train \
    --val_list test \
    --test_list test \
    --depth 50 \
    --resume_weights ./_prev_20220726/weights/imagenet/resnet50-11ad3fa6.pth \
    --epochs 1000 \
    --lr 1e-2 \
    --weight_decay 5e-4 \
    --image_cut crop \
    --loss_function CrossEntropyLoss \
    --fgssl True \
    --patches 16 8 4 \
    --gpus ${GPU} \
    --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log 


# model_name
## cifar imagenet CNN

# loss_function
## CrossEntropyLoss LogitAdjust FocalLoss

# image_cut
## crop resize

# Pretext task
## lorotE lorotI