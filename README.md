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
- DINO v2 linear probing 실험 아래 코드로 실행 가능
```bash
  bash scripts/dinov2.sh
```
1. **Linear Probing 실험 파라미터**

   해당 파일을 실행하면 가능한 모든 조합에 대해 실험이 진행됩니다.

   - **Learning Rate**  
     Default:  
     `{0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}`  
     예시:  
     ```bash
     --lr_list 0.2, 0.3, 0.5
     ```

   - **DINO의 Output Block 개수 (뒤에서부터)**  
     Default:  
     `{1, 4}`  
     예시:  
     ```bash
     --last_blocks_list 1 2
     ```

   - **Average-pooled Patch Token 사용 여부**  
     - **Yes**: Class Token과 Average-pooled Patch Token을 함께 사용  
     - **No**: Class Token만 사용
     Default:  
     `{False, True}`  
     예시:  
     ```bash
     --avg_pool_list False
     ```

2. **Concat Augmentation**

   `concat` augmentation을 사용하려면, shell 파일 실행 시 `--concat` 인자를 추가하면 됩니다.
  
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
