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
        exp_name1 = '13_IMGN50_512_Pre_CE_resize/13_IMGN50_512_Pre_CE_resize.pth' # base R50 w/pt BSCE
        exp_name2 = '14_IMGN50_512_Pre_CE_crop/14_IMGN50_512_Pre_CE_crop.pth' # lorot i
        exp_name3 = '15_IMGN50_512_Pre_BS_resize/15_IMGN50_512_Pre_BS_resize.pth' # lorot e
        exp_name4 = '16_IMGN50_512_Pre_BS_crop/16_IMGN50_512_Pre_BS_crop.pth'  # lorot e

        load_dir1 = "./ckpts/{}".format(exp_name1)
        load_dir2 = "./ckpts/{}".format(exp_name2)
        load_dir3 = "./ckpts/{}".format(exp_name3)
        load_dir4 = "./ckpts/{}".format(exp_name4)

        # dataset
        print(">> Load the data...", flush=True)
        val_dataset = FSTDataset('../data/251212_G2_Rev_Accumulate/'+ args.test_list + '_' + str(i) +'.json', is_train=False, augment=['flip'], crop=True, model_type=args.model_name)
        print("Test: {:d}".format(len(val_dataset)), flush=True)

        # dataloader
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, drop_last=False)
        print("<< Finished.\n", flush=True)

        # model
        print(">> Load the model...", flush=True)
        if args.model_name == 'cifar' or args.model_name == 'imagenet':
            model1 = RN.ResNet(args.model_name, args.depth, args.num_classes, args.bottleneck)
            model2 = RN.ResNet(args.model_name, args.depth, args.num_classes, args.bottleneck, args.pretext)
            model3 = RN.ResNet(args.model_name, args.depth, args.num_classes, args.bottleneck, args.pretext)

        path1 = os.path.join(load_dir1)
        path2 = os.path.join(load_dir2)
        path3 = os.path.join(load_dir3)
        if os.path.isfile(path1): ## eval
            checkpoint = torch.load(path1)
            model1.load_state_dict(checkpoint['state_dict'],strict=False)
            print("=> loaded weight '{}'".format(path1))

        if os.path.isfile(path2): ## eval
            checkpoint = torch.load(path2)
            model2.load_state_dict(checkpoint['state_dict'],strict=False)
            print("=> loaded weight '{}'".format(path2))

        if os.path.isfile(path3): ## eval
            checkpoint = torch.load(path3)
            model3.load_state_dict(checkpoint['state_dict'],strict=False)
            print("=> loaded weight '{}'".format(path3))


        model1 = model1.cuda()
        model2 = model2.cuda()
        model3 = model3.cuda()
        print("<< Model Loaded.\n")

        result1 = val(args, val_loader, model1)
        result2 = val(args, val_loader, model2)
        result3 = val(args, val_loader, model3)
        print("\n1 : Single Model Validation \t Val Acc {:.3f}".format(result1['total_acc']), flush=True)
        print("Class Acc {}".format(result1['class_acc']), flush=True)

        print("\n2 : Single Model Validation \t Val Acc {:.3f}".format(result2['total_acc']), flush=True)
        print("Class Acc {}".format(result2['class_acc']), flush=True)

        print("\n3 : Single Model Validation \t Val Acc {:.3f}".format(result3['total_acc']), flush=True)
        print("Class Acc {}".format(result3['class_acc']), flush=True)

        result_all = val_ensemble(args, val_loader, model1, model2, model3)
        print("\nALL : Ensemble Validation \t Val Acc {:.3f}".format(result_all['total_acc']), flush=True)
        print("Class Acc {}".format(result_all['class_acc']), flush=True)


        print("------------------------------------------------------------------------")


        print("<< Finished.\n")
        return

def val_ensemble(args, val_loader, model1, model2, model3):
    model1.eval()
    model2.eval()
    model3.eval()
    n_image = len(val_loader.dataset)
    n_batch = len(val_loader)
    bs = args.batch_size
    total_preds = torch.zeros(n_image)
    total_labels = torch.zeros(n_image)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()

            logits1, _ = model1(images)
            logits2, _ = model2(images)
            logits3, _ = model3(images)
            T = 1
            logits = torch.softmax(logits2 ** T, dim=1) + torch.softmax(logits3 ** T, dim=1)
            # logits = torch.softmax(logits1 ** T, dim=1) + torch.softmax(logits2 ** T, dim=1) + torch.softmax(logits3 ** T, dim=1)
            # logits = logits1 + logits2 + logits3
            _, preds = torch.max(logits, 1)
            total_idx = bs * batch_idx
            if batch_idx == n_batch-1:
                total_preds[total_idx:] = preds.detach().cpu()
                total_labels[total_idx:] = labels.detach().cpu()
            else:
                total_preds[total_idx:total_idx+bs] = preds.detach().cpu()
                total_labels[total_idx:total_idx+bs] = labels.detach().cpu()

    total_acc = (total_labels == total_preds).sum() / n_image

    matrix = confusion_matrix(total_labels, total_preds)
    acc_list = matrix.diagonal()/matrix.sum(axis=1)
    acc_list = [round(x*100, 2) for x in acc_list]

    # plot confusion

    plot_confusion_matrix(matrix, class_name, './img/{}_confusion_matrix.png'.format(args.exp_name))

    return {
            'total_acc': total_acc.item() * 100,
            'class_acc': acc_list}

def val(args, val_loader, model):

    model.eval()
    n_image = len(val_loader.dataset)
    print(n_image)
    n_batch = len(val_loader)
    bs = args.batch_size
    total_preds = torch.zeros(n_image)
    total_labels = torch.zeros(n_image)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()

            logits, _ = model(images)

            _, preds = torch.max(logits, 1)
            total_idx = bs * batch_idx
            if batch_idx == n_batch-1:
                total_preds[total_idx:] = preds.detach().cpu()
                total_labels[total_idx:] = labels.detach().cpu()
            else:
                total_preds[total_idx:total_idx+bs] = preds.detach().cpu()
                total_labels[total_idx:total_idx+bs] = labels.detach().cpu()

    total_acc = (total_labels == total_preds).sum() / n_image
    matrix = confusion_matrix(total_labels, total_preds)
    acc_list = matrix.diagonal()/matrix.sum(axis=1)
    acc_list = [round(x*100, 2) for x in acc_list]

    # plot confusion

    # plot_confusion_matrix(matrix, class_name, './img/{}_confusion_matrix.png'.format(args.exp_name))

    return {'total_acc': total_acc.item() * 100,
            'class_acc': acc_list}

def val_nce(args, val_loader, model, criterion, classifier=None):

    model.eval()
    n_image = len(val_loader.dataset)
    n_batch = len(val_loader)
    bs = args.batch_size
    total_loss = 0.
    total_preds = torch.zeros(n_image)
    total_labels = torch.zeros(n_image)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()

            feats = model(images, nce=True, val=True)
            logits = classifier(feats)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            total_idx = bs * batch_idx
            if batch_idx == n_batch-1:
                total_preds[total_idx:] = preds.detach().cpu()
                total_labels[total_idx:] = labels.detach().cpu()
            else:
                total_preds[total_idx:total_idx+bs] = preds.detach().cpu()
                total_labels[total_idx:total_idx+bs] = labels.detach().cpu()

    total_acc = (total_labels == total_preds).sum() / n_image
    total_loss /= n_image
    matrix = confusion_matrix(total_labels, total_preds)
    acc_list = matrix.diagonal()/matrix.sum(axis=1)
    acc_list = [round(x*100, 2) for x in acc_list]

    # plot confusion

    plot_confusion_matrix(matrix, class_name, './img/{}_confusion_matrix.png'.format(args.exp_name))

    return {'loss': total_loss,
            'total_acc': total_acc.item() * 100,
            'class_acc': acc_list}

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    torch.autograd.set_detect_anomaly(True)
    args = get_opts()
    print(json.dumps(vars(args), indent='\t'), flush=True)
    main(args)