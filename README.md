# 2024.12 FST Codebase
# Environment setting

### Data 경로 설정
현재 디렉토리 path 가 . 라고 가정하였을시에
../data에 데이터 추가. 
```
.
├── 2412_v4_integrated
├── data
    └── 251212_G2_Rev_Accumulate_v3
        └── 00_Normal 
        └── 01_Spot    
        └── 03_Solid
        └── 04_Dark Dust
        └── 05_Shell
        └── 06_Fiber
        └── 07_Smear
        └── 08_Pinhole
        └── 11_OutFo   
        └── train.json
        └── test.json
```

### Package 설치.
1) 권장 Docker 사용시 (lakepark/fst:latest).
```aiignore
docker run \
	-it \
	--shm-size 32G \
	-v 마운트경로:/project  \ 
	--gpus '"device=0,1"' \ 
	--name FST \
	lakepark/fst:latest \
	/bin/bash
```
2) 다른 환경 사용시. 필요 패키지는 requirements.txt 참고. 설치 명령어는 아래.
```aiignore
pip -r requirements.txt
```

# Training 학습.
scripts 안에 있는 shell 파일 참고해서 EXP_NAME 및 파라미터 수정 후 실행.
- EXP_NAME : 실험번호_실험명_하이퍼파라미터
```
ex) EXP_NAME=1_CNN3_512_BS_resize
```

## Baseline
빠른 실행 명령어
```bash
  bash scripts/13_IMGN50_512_Pre_CE_resize.sh
  bash scripts/14_IMGN50_512_Pre_CE_crop.sh
  bash scripts/15_IMGN50_512_Pre_BS_resize.sh
  bash scripts/16_IMGN50_512_Pre_BS_crop.sh
```
- Loss Function : CE / BS 는 Cross-Entropy / Binary Softmax 를 의미. CE가 일반적 Classification Loss / BS가 데이터 적은 클래스에 대해서도 강건한 학습을 진행하는 Loss. 
- Augmentation : Resize / Crop

위의 option에 따라 13~16번 중 Shell 파일을 선택하여 실행하면 됩니다.

## DINOv2 linear probing 
빠른 실행 명령어
```bash
  bash scripts/dinov2.sh
```
<details>
<summary>Details</summary>
** Linear Probing 실험 파라미터 **

   해당 dinov2 쉘 파일에서 다음과 같은 파라미터들을 아래와 같이 수정하면 가능한 모든 조합에 대해 실험이 진행됩니다.

   - **model_name** : DINOv2 아키텍처 선택    
     Default:
     dinov2_vits14  
     선택 가능 후보군 `{dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14, dinov2_vits14_reg, dinov2_vitb14_reg, dinov2_vitl14_reg, dinov2_vitg14_reg}`
     예시:  
     ```bash
     --model_name dinov2_vits14
     ```

   - **Learning Rate** : Linear Probing 학습 시 사용할 Learning Rate
     예시:  
     ```bash
     --lr_list 0.2, 0.3, 0.5
     ```

   - **DINO의 Output Block 개수 (뒤에서부터)** : Linear Probing에 Multi-scale feature를 어느정도로 사용할지에 대한 옵션  
     Default:  
     `{1, 4}`  
     예시:  (아래와 같이 실행하면, block 1개 / 2개로 돌리는 두가지 실험이 진행됩니다.)
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
     
   - **Concat Augmentation** : Image Resize / Crop 각각 사용시 각각 장단점이 있으므로, Concat Augmentation을 사용하면 두 가지 augmentation으로 모두 중간 결과값을 얻어 합쳐서 최종 결과값을 도출하는 방법입니다. 

   `concat` augmentation을 사용하려면, shell 파일 실행 시 `--concat` 인자를 추가하면 됩니다.


</details>



## Fine-grained Self-Supervision
빠른 실행 명령어
```bash 
  bash scripts/20_IMGN50_512_Pre_CE_crop_fgssl.sh
```

<details>
<summary>Details</summary>
**Fine-Grained SSL**

   `fgssl` 학습을 하려면, shell 파일 실행 시 `--fgssl` 인자를 추가하면 됩니다.

   - **Patch size**  
     학습 시 patch의 크기를 설정할 수 있습니다. 복수개 사용 가능.
     예시:  
     ```bash
     --patches 16 8 4
     ```

   - **Classifier dimension**  
     각 Block의 Classifier의 dimension을 설정할 수 있습니다.
     Default:  
     `512`  
     예시:  
     ```bash
     --featdim 512
     ```

</details>

## Patch-wize image concatenation
빠른 실행 명령어
```bash 
  bash scripts/21_IMGN50_512_Pre_BS_patch.sh
```

<details>
<summary>Details</summary>

   - **patch_num**  
     패치의 개수를 조절하는 argument입니다. 64로 설정하면 8X8 모양의 패치를 만들어 이미지를 concat합니다.
     ```bash
     --patches 64
     ```

</details>