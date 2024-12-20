import os
import warnings
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from loss.logitadjust_balanced_softmax import *
from loss.Focalloss_ldamloss import *
from loss.contrastive import SupConLoss
from loss.balanced_contrastive import BalSCL

from utils.opt import get_opts
from utils.pretext import lorotE, lorotI, rotation
from dataset.datasets import FSTDataset
import models.resnet as RN
import models.CNN_models as CNN
import sys

from visualization.plot_confusion import plot_confusion_matrix
import shutil

class_name = ['Norm', 'BrightDot', 'DarkDot', 'Defocus', 'FacingForeign', 'Hole', 'OpposingForeign', 'RingDot', 'Unique']

def main(args):
    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    print("\nCUDA_VISIBLE_DEVICES: [{}]\n".format(os.environ["CUDA_VISIBLE_DEVICES"]), flush=True)
    
    # seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ensemble checkpoint
    split_len = 5
    for i in range(split_len):
        # model checkpoints
        exp_names = [
            f'Split_{i}_13_IMGN50_512_Pre_CE_resize/Split_{i}_13_IMGN50_512_Pre_CE_resize.pth', # base R50 w/pt BSCE
            f'Split_{i}_14_IMGN50_512_Pre_CE_crop/Split_{i}_14_IMGN50_512_Pre_CE_crop.pth', # lorot i
            f'Split_{i}_15_IMGN50_512_Pre_BS_resize/Split_{i}_15_IMGN50_512_Pre_BS_resize.pth', # lorot e
            f'Split_{i}_16_IMGN50_512_Pre_BS_crop/Split_{i}_16_IMGN50_512_Pre_BS_crop.pth',  # lorot e
        ]

        load_dirs = ["./ckpts/{}".format(exp_name) for exp_name in exp_names]
        models = []
        for i, load_dir in enumerate(load_dirs):
            model = RN.ResNet(args.model_name, args.depth, args.num_classes, args.bottleneck,
                              getattr(args, 'pretext', None))

            # load checkpoints
            if os.path.isfile(load_dir):
                checkpoint = torch.load(load_dir)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print(f"=> loaded weight '{load_dir}'")
            else:
                print(f"=> no checkpoint found at '{load_dir}'")
            models.append(model.cuda())
        print("<< Model Loaded.\n")

        # dataset
        print(">> Load the data...", flush=True)
        val_dataset = FSTDataset('../data/251212_G2_Rev_Accumulate_v2/'+ args.test_list + '_' + str(i) +'.json', is_train=False, augment=['flip'], crop=True, model_type=args.model_name)
        print("Val: {:d}".format(len(val_dataset)), flush=True)

        # dataloader
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False)
        print("<< Dataloading Finished.\n", flush=True)

        # Save the visualized wrong predictions
        wrongpred_dir = './img/wrong_pred'
        os.makedirs(wrongpred_dir, exist_ok=True)

        model.eval()
        class_folder_list = val_dataset.class_list

        with torch.no_grad():
            for batch_idx, (images, names, labels) in enumerate(val_loader):  # labels는 실제 라벨
                images = images.cuda()

                # 모델별 로짓 및 예측 계산
                logits_list = [model(images)[0] for model in models]
                preds_list = [torch.max(torch.softmax(logits, dim=1), 1).indices for logits in logits_list]
                max_probs_list = [torch.max(torch.softmax(logits, dim=1), 1).values for logits in logits_list]
                # print(names)
                for i, name in enumerate(names):
                    set_n = name.split('/')[-2]  # '00_Normal'
                    img_n = name.split('/')[-1]  # 'RAW_2 ADR_DF_BTM RV_130.bmp'

                    cnt = 0
                    for model_idx, (preds, max_probs) in enumerate(zip(preds_list, max_probs_list)):
                        pred_label = preds[i]
                        true_label = labels[i]  # 실제 라벨

                        if pred_label != true_label:  # 잘못 예측한 경우
                            cls_n = class_folder_list[pred_label]  # 예측 클래스 이름
                            conf_val = math.floor(max_probs[i].item() * 10) / 10  # 확률 값 반올림

                            # 모델별 잘못된 예측 폴더 구조
                            cp_path = os.path.join(wrongpred_dir, f"model_{str(model_idx)}", str(set_n), str(cls_n), str(conf_val))
                            os.makedirs(cp_path, exist_ok=True)

                            # 잘못 예측된 이미지 복사
                            shutil.copy2(name, os.path.join(cp_path, img_n))
                            cnt += 1
                    if cnt > 1:
                        # 모델별 잘못된 예측 폴더 구조
                        multiple_path = os.path.join(wrongpred_dir, f"model_multiple_wrong_{str(cnt)}", str(set_n))
                        if not os.path.exists(multiple_path):
                            os.makedirs(multiple_path, exist_ok=True)

                        # 잘못 예측된 이미지 복사
                        shutil.copy2(name, os.path.join(multiple_path, img_n))


        print("<< Finished!")



if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    torch.autograd.set_detect_anomaly(True)
    args = get_opts()
    print(json.dumps(vars(args), indent='\t'), flush=True)
    main(args)