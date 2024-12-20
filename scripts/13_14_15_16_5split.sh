GPU=0
train_lists=("train_0" "train_1" "train_2" "train_3" "train_4")
test_lists=("test_0" "test_1" "test_2" "test_3" "test_4")

for i in {0..4}; do
    train_list="${train_lists[i]}"
    test_list="${test_lists[i]}"
    echo "Running with train_list=${train_list} and test_list=${test_list}"

    EXP_NAME="Split_${i}_13_IMGN50_512_Pre_CE_resize"
    python train.py \
        --model_name resnet \
        --train_list "${train_list}" \
        --test_list "${test_list}" \
        --val_list "${test_list}" \
        --depth 50 \
        --resume_weights /mnt/hdd0/FST/_prev_20220726/weights/imagenet/resnet50-11ad3fa6.pth \
        --loss_function CrossEntropyLoss \
        --image_cut resize \
        --gpu ${GPU} \
        --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log

    EXP_NAME="Split_${i}_14_IMGN50_512_Pre_CE_crop"
    python train.py \
        --model_name resnet \
        --train_list "${train_list}" \
        --test_list "${test_list}" \
        --val_list "${test_list}" \
        --depth 50 \
        --resume_weights /mnt/hdd0/FST/_prev_20220726/weights/imagenet/resnet50-11ad3fa6.pth \
        --loss_function CrossEntropyLoss \
        --image_cut crop \
        --gpu ${GPU} \
        --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log

    EXP_NAME="Split_${i}_15_IMGN50_512_Pre_BS_resize"
    python train.py \
        --model_name resnet \
        --train_list "${train_list}" \
        --test_list "${test_list}" \
        --val_list "${test_list}" \
        --depth 50 \
        --resume_weights /mnt/hdd0/FST/_prev_20220726/weights/imagenet/resnet50-11ad3fa6.pth \
        --loss_function LogitAdjust \
        --image_cut resize \
        --gpu ${GPU} \
        --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log

    EXP_NAME="Split_${i}_16_IMGN50_512_Pre_BS_crop"
    python train.py \
        --model_name resnet \
        --train_list "${train_list}" \
        --test_list "${test_list}" \
        --val_list "${test_list}" \
        --depth 50 \
        --resume_weights /mnt/hdd0/FST/_prev_20220726/weights/imagenet/resnet50-11ad3fa6.pth \
        --loss_function LogitAdjust \
        --image_cut crop \
        --gpu ${GPU} \
        --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}.log
done