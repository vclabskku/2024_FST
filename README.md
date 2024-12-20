# 2024.12 FST Codebase
## Environment setting
### Data

```
ln -s /mnt/hdd0/FST/data ./
```


pip install seaborn

## How to use
scripts 안에 있는 shell 파일 참고해서 EXP_NAME 및 파라미터 수정 후 실행.
- EXP_NAME : 실험번호_실험명_하이퍼파라미터
```
ex) EXP_NAME=1_CNN3_512_BS_resize
```

### About Arguments (For more details, check ./utils/opt.py)
--model_name 
    CNN :
    cifar : 
    imagenet : resnet

--num_layer 
    3 : number of middle layers in CNN (cifar & imagenet은 해상사항 X)

--hidden_dim 
    512 : embedding dimension of CNN

--loss_function
    CrossEntropyLoss
    FocalLoss
    LogitAdjust 

--image_cut
    crop
    resize

--train_list -> list of data to be used for train
    train 


--val_list -> list of data to be used for validation
    val
