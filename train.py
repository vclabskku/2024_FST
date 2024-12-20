import os
import warnings
import json
import random
import sys

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

from visualization.plot_confusion import plot_confusion_matrix
from collections import OrderedDict

class_name = ['00_Normal', '01_Spot', '03_Solid', '04_Dark Dust', '05_Shell', '06_Fiber', '07_Smear', '08_Pinhole', '11_OutFo']

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

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

    # checkpoint
    save_dir = "./ckpts/{}".format(args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- dataset
    print(">> Load the data...", flush=True)
    train_dataset = FSTDataset(args, args.dataset_dir + args.train_list +'.json', is_train=True, augment=['flip'], crop=True, model_type=args.model_name)
    val_dataset = FSTDataset(args, args.dataset_dir + args.test_list +'.json', is_train=False, augment=['flip'], crop=True, model_type=args.model_name)
    print("Train: {:d} / Val: {:d}".format(len(train_dataset), len(val_dataset)), flush=True)
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    args.num_classes = train_dataset.num_classes
    print("<< Data Loading Finished.\n", flush=True)

    # model
    print(">> Load the model...", flush=True)
    if args.model_name == 'resnet':
        model = RN.ResNet(args.model_name, args.depth, args.num_classes, args.bottleneck, args.pretext)
    elif 'ViT' in args.model_name:
        # if add more model, add here
        pass
    else:
        raise Exception('unknown network architecture: {}'.format(args.model_name))

    num_params = count_parameters(model)
    print("Total Parameter: \t%2.1fM" % num_params)

    # --- classifier layer for contrastive
    if 'contrastive' in args.loss_function:
        classifier = nn.Linear(model.fc_in_dim, args.num_classes).cuda()
        classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        classifier = None

    if args.resume_weights:
        if 'ViT' in args.model_name: # if add more model, add here
            pass
        else:
            path = os.path.join(args.resume_weights)
            if os.path.isfile(path):
                checkpoint = torch.load(path)
                post_checkpoint = {}
                for key, value in checkpoint.items():
                    if 'fc.weight' == key or 'fc.bias' == key:
                        continue
                    if value.dim() == 2 and model.state_dict()[key].dim() == 4:
                        post_checkpoint[key] = value.unsqueeze(-1).unsqueeze(-1)
                    else:
                        post_checkpoint[key] = value
                model.load_state_dict(post_checkpoint,strict=False)
                print("=> loaded weight '{}'".format(path))
    else:   
        print("Model: {} from scratch".format(args.model_name))

    model = model.cuda()
    cls_num_list = [578, 3947, 155, 618, 10, 574, 39, 3, 242] # v3
    print("<< Model Loading Finished.\n")

    # --- loss function --- #
    criterion_ce = nn.CrossEntropyLoss()
    BCL_criterion = None

    loss_split = args.loss_function.split('+')
    args.loss_function = loss_split[0]
    loss_function2 = None
    if len(loss_split) != 1:
        loss_function2 = loss_split[1]
    # loss function
    if args.loss_function == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_function == 'LogitAdjust':
        criterion = LogitAdjust(cls_num_list)
    elif args.loss_function == 'FocalLoss':
        criterion = FocalLoss()
    elif args.loss_function == 'contrastive':
        BCL_criterion = SupConLoss()
        if loss_function2 is not None:
            if loss_function2 == 'LogitAdjust':
                criterion = LogitAdjust(cls_num_list)
            elif loss_function2 == 'CrossEntropyLoss':
                criterion = nn.CrossEntropyLoss()
        else:  ## default w/ LogitAdjust
            criterion = LogitAdjust(cls_num_list)
    elif args.loss_function == 'BalSCL':
        BCL_criterion = BalSCL(cls_num_list, 0.07)
        if loss_function2 is not None:
            if loss_function2 == 'LogitAdjust':
                criterion = LogitAdjust(cls_num_list)
            elif loss_function2 == 'CrossEntropyLoss':
                criterion = nn.CrossEntropyLoss()
        else: ## default w/ LogitAdjust
            criterion = LogitAdjust(cls_num_list)
    if args.loss_function == 'BalSCL':
        precriterion = nn.CrossEntropyLoss()
    elif args.loss_function == 'contrastive':
        precriterion = nn.CrossEntropyLoss()
    else:
        precriterion = nn.CrossEntropyLoss()
    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=len(train_loader)*args.step_size, 
                                                gamma=args.gamma)

    # train
    print(">> Train...", flush=True)
    best_acc = 0.
    best_epoch = 0
    best_class_acc = None
    for epoch in range(args.epochs):
        is_best=False
        if args.loss_function == 'contrastive':
            train_nce(args, train_loader, model, BCL_criterion, precriterion, optimizer, scheduler, epoch, criterion, classifier_optimizer, classifier)
            result = val_nce(args, val_loader, model, criterion_ce, classifier)
        else:
            train(args, train_loader, model, criterion, precriterion, optimizer, scheduler, epoch, BCL_criterion=BCL_criterion)
            result = val(args, val_loader, model, criterion)
        print("\nEpoch [{:03d}/{:03d}]\tVal Loss {:.3f}\t Val Acc {:.3f}"\
                .format(epoch+1, args.epochs, result['loss'], result['total_acc']), flush=True)
        print("Class Acc {}".format(result['class_acc']), flush=True)

        is_best = True
        best_epoch = epoch + 1
        best_acc = max(best_acc, result['total_acc'])
        best_class_acc = result['class_acc']

        print("------------------------------------------------------------------------")
        model_filename = os.path.join(save_dir, f'{args.exp_name}.pth')
        if epoch > args.epochs // 2:
            if is_best:
                torch.save(
                    {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    },
                    model_filename
                )
    print("\nBest Epoch {:03d} \t Best Acc {:.3f}" \
          .format(best_epoch, best_acc), flush=True)
    print("Best Class Acc {}".format(best_class_acc), flush=True)
    print("<< Finished.\n")
    return


def train(args, train_loader, model, criterion, precriterion, optimizer, scheduler, epoch, BCL_criterion=None):
    model.train()
    n_batch = len(train_loader)
    bs = args.batch_size
    for batch_idx, (images, names, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        if args.pretext == 'lorotE':
            images, rotlabel = lorotE(images)
        elif args.pretext == 'lorotI':
            images, rotlabel = lorotI(images)
        if args.model_name == 'CNN':
            logits, plogits = model(images, return_feature=False, layers=None, input_layers=None, onlyfc=False)
        else:
            if args.loss_function == 'BalSCL':
                images1, _ = rotation(images)
                images1, rotlabel_e = lorotE(images1.detach().clone())
                images2, _ = rotation(images)
                images2, rotlabel_i = lorotI(images2.detach().clone())
                logits1, plogits1, feats1, pfeats1, centers1 = model(images1, BCL=True)
                logits2, plogits2, feats2, pfeats2, centers2 = model(images2, BCL=True)
                loss = criterion(logits1, labels) + criterion(logits2, labels)
            else:
                logits, plogits = model(images)
                loss = criterion(logits, labels)
        if args.loss_function == 'BalSCL':
            feats = torch.cat([feats1.unsqueeze(1), feats2.unsqueeze(1)], dim=1)
            scl_loss = BCL_criterion(centers1, feats, labels)
            loss = args.alpha * loss + args.beta * scl_loss
            if args.pretext == 'lorotE' or args.pretext == 'lorotI':
                loss = loss + args.pretext_ratio * precriterion(plogits1, rotlabel_e)
                loss = loss + args.pretext_ratio * precriterion(plogits2, rotlabel_i)
        else:
            if args.pretext == 'lorotE' or args.pretext == 'lorotI':
                loss = loss + args.pretext_ratio * precriterion(plogits, rotlabel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        sys.stdout.write('\r')
        sys.stdout.write("Epoch [{:02d}/{:02d}]\tBatch [{:02d}/{:02d}]\tTrain Loss {:.3f}"\
                .format(epoch+1, args.epochs, batch_idx+1, n_batch, loss.item()/bs))
        sys.stdout.flush()
    return

def train_nce(args, train_loader, model, criterion, precriterion, optimizer, scheduler, epoch, criterion_ce, classifier_optimizer, classifier=None):
    model.train()
    n_batch = len(train_loader)
    bs = args.batch_size

    for batch_idx, (images, names, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        images1,_ = rotation(images)
        images1, rotlabel_e = lorotE(images1.detach().clone())
        images2,_ = rotation(images)
        images2, rotlabel_i = lorotI(images2.detach().clone())

        logits1, plogits1, plogits_feat1 = model(images1, nce=True)
        logits2, plogits2, plogits_feat2 = model(images2, nce=True)

        features = torch.cat([logits1.unsqueeze(1), logits2.unsqueeze(1)], dim=1)
        # pretext_features = torch.cat([plogits1.unsqueeze(1), plogits2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels)


        if args.pretext == 'lorotE' or args.pretext == 'lorotI':
            loss += args.pretext_ratio * precriterion(plogits_feat1, rotlabel_e)
            loss += args.pretext_ratio * precriterion(plogits_feat2, rotlabel_i)
            # loss += args.pretext_ratio * precriterion(pretext_features, rotlabel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        sys.stdout.write('\r')
        sys.stdout.write("Epoch [{:02d}/{:02d}]\tBatch [{:02d}/{:02d}]\tTrain Loss {:.3f}" \
                         .format(epoch + 1, args.epochs, batch_idx + 1, n_batch, loss.item() / bs))
        sys.stdout.flush()

    for batch_idx, (images, names, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()


        feats = model(images, nce=True, val=True).detach().clone()
        logits = classifier(feats)

        classifier_loss = criterion_ce(logits, labels)

        classifier_optimizer.zero_grad()
        classifier_loss.backward()
        classifier_optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write("Epoch [{:02d}/{:02d}]\tBatch [{:02d}/{:02d}]\tTrain Loss {:.3f}" \
                         .format(epoch + 1, args.epochs, batch_idx + 1, n_batch, loss.item() / bs))
        sys.stdout.flush()
        # print("Epoch [{:02d}/{:02d}]\tBatch [{:02d}/{:02d}]\tTrain Loss {:.3f}"\
        #         .format(epoch+1, args.epochs, batch_idx+1, n_batch, loss.item()/bs), flush=True)

    return

def val(args, val_loader, model, criterion):

    model.eval()
    n_image = len(val_loader.dataset)
    n_batch = len(val_loader)
    bs = args.batch_size
    total_loss = 0.
    total_preds = torch.zeros(n_image)
    total_labels = torch.zeros(n_image)

    with torch.no_grad():
        for batch_idx, (images, names, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()

            logits, _ = model(images)
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

    plot_confusion_matrix(matrix, class_name, './img/{}_confusion_matrix_ACC_{:.3f}.png'.format(args.exp_name, total_acc.item() * 100))

    return {'loss': total_loss,
            'total_acc': total_acc.item() * 100,
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
        for batch_idx, (images, names, labels) in enumerate(val_loader):
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