import os
import warnings
import json
import random
import sys
from tqdm import tqdm

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
from dataset.datasets import FSTDataset, barlowDataset
import models.resnet as RN
from models.dino import Create_DINO
from models.pmg import PMG
from utils.fgssl import get_ce_optimizer, get_barlow_optimizer, jigsaw_generator, barlow_criterion, cosine_anneal_schedule


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
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    # fg-ssl unlabel dataloader
    if args.fgssl:
        unlabel_dataset = barlowDataset(args, args.dataset_dir + args.train_list + '.json')
        unlabel_dataloader = torch.utils.data.DataLoader(unlabel_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    print("<< Data Loading Finished.\n", flush=True)

    # model
    print(">> Load the model...", flush=True)
    if args.model_name == 'resnet':
        model = RN.ResNet(args.model_name, args.depth, args.num_classes, args.bottleneck, args.pretext)
    
    elif args.model_name == 'resnet':
        model = RN.ResNet(args.model_name, args.depth, args.num_classes, args.bottleneck, args.pretext)
    elif args.model_name == 'ResNet_patch16':
        model = RN.ResNet_patch16(args.model_name, args.depth, args.num_classes, args.bottleneck, args.pretext, patch_num=args.patch_num)
    elif 'ViT' in args.model_name:
        # if add more model, add here
        pass
    elif args.model_name.startswith('dinov2'):
        # 'dinov2_vits14' / 'dinov2_vitb14' / 'dinov2_vitl14' / 'dinov2_vitg14' / 'dinov2_vits14_reg' / 'dinov2_vitb14_reg' / 'dinov2_vitl14_reg' / 'dinov2_vitg14_reg'
        if args.concat:
            sample = train_dataset[0][0][0].unsqueeze(0)
        else:
            sample = train_dataset[0][0].unsqueeze(0)
        model = Create_DINO(args.model_name, sample, args.batch_size, args.concat, args.num_classes)
    else:
        raise Exception('unknown network architecture: {}'.format(args.model_name))

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

    # fg-ssl get pmg model
    if args.fgssl:
        if args.model_name != 'resnet' or args.depth != 50 or args.resume_weights is None:
            raise Exception('FG-SSL needs pretrained resnet50 now.')
        for param in model.parameters():
            param.requires_grad = True
        model = PMG(args, model, args.featdim, args.num_classes)

    num_params = count_parameters(model)
    print("Total Parameter: \t%2.1fM" % num_params)

    model = model.cuda()
    cls_num_list = train_dataset.cls_num_list
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
    if not args.fgssl:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=len(train_loader)*args.step_size, 
                                                    gamma=args.gamma)
    elif args.model_name.startswith('dinov2'):
        optimizer = torch.optim.SGD(model.optim_param_groups, momentum=0.9, weight_decay=0)
        max_iter = 10 * 1250
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)   

    else:
        barlow_optimizer = get_barlow_optimizer(args, model)
        ce_optimizer = get_ce_optimizer(args, model)

    # train
    print(">> Train...", flush=True)
    best_acc = 0.
    best_epoch = 0
    best_class_acc = None
    is_best = False
    for epoch in range(args.epochs):
        is_best=False
        if args.loss_function == 'contrastive':
            train_nce(args, train_loader, model, BCL_criterion, precriterion, optimizer, scheduler, epoch, criterion, classifier_optimizer, classifier)
            result = val_nce(args, val_loader, model, criterion_ce, classifier)
        elif args.fgssl:
            # half epoch for training jigsaw
            if epoch < args.epochs // 2:
                ssl_loss = train_barlow(args, model, unlabel_dataloader, barlow_optimizer, epoch)
                continue
            # other half epoch for fine-tuning
            else:
                losses = train_fine(args, model, train_loader, ce_optimizer, criterion, epoch)
                result = val_fine(args, model, val_loader, criterion)
        else:
            train(args, train_loader, model, criterion, precriterion, optimizer, scheduler, epoch, BCL_criterion=BCL_criterion)
            if args.model_name.startswith('dinov2'):
                result, best_key = val_dino(args, val_loader, model, criterion)
            else:
                result = val(args, val_loader, model, criterion)

        print("\nEpoch [{:03d}/{:03d}]\tVal Loss {:.3f}\t Val Acc {:.3f}"\
                .format(epoch+1, args.epochs, result['loss'], result['total_acc']), flush=True)
        print("Class Acc {}".format(result['class_acc']), flush=True)

        if best_acc < result['total_acc']:
            is_best = True
            best_epoch = epoch + 1
            best_acc = max(best_acc, result['total_acc'])
            best_class_acc = result['class_acc']

        print("------------------------------------------------------------------------")
        model_filename = os.path.join(save_dir, f'{args.exp_name}.pth')
        if epoch > args.epochs // 2:
            if is_best:
                if args.model_name.startswith('dinov2'):
                    torch.save(
                    {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_linear' : best_key
                    },
                    model_filename
                )
                else:
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
        if not args.concat:
            images = images.cuda()
        labels = labels.cuda()
        if args.pretext == 'lorotE':
            images, rotlabel = lorotE(images)
        elif args.pretext == 'lorotI':
            images, rotlabel = lorotI(images)
        if args.model_name == 'CNN':
            logits, plogits = model(images, return_feature=False, layers=None, input_layers=None, onlyfc=False)
        elif args.model_name.startswith('dinov2'):
            logits = model(images)
            losses = {f"loss_{k}": criterion(v, labels) for k, v in logits.items()}
            loss = sum(losses.values())
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

def train_barlow(args, model, unlabel_dl, optimizer, epoch):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    # lr = [x * 5.0 for x in lr]
    losses = 0

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer.zero_grad()
    for img_1, table_1_img, table_2_img, table_3_img, img_2 in tqdm(unlabel_dl, desc = "barlowtwins train", position = 1, leave = False):
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = lr[nlr] #cosine_anneal_schedule(epoch, args.epochs, lr[nlr])    
        img_1, table_1_img, table_2_img, table_3_img, img_2 = img_1.float().to(args.device), table_1_img.float().to(args.device), table_2_img.float().to(args.device), table_3_img.float().to(args.device), img_2.float().to(args.device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            _, _, _, _, x_ft_1, _, _, _ = model(table_1_img)
            _, _, _, _, y_ft_1, _, _, _ = model(img_2)
            barlow_loss_1 = barlow_criterion(x_ft_1, y_ft_1)
        scaler.scale(barlow_loss_1).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += barlow_loss_1

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            _, _, _, _, _, x_ft_2, _, _ = model(table_2_img)
            _, _, _, _, _, y_ft_2, _, _ = model(img_2)
            barlow_loss_2 = barlow_criterion(x_ft_2, y_ft_2)
        scaler.scale(barlow_loss_2).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += barlow_loss_2

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            _, _, _, _, _, _, x_ft_3, _ = model(table_3_img)
            _, _, _, _, _, _, y_ft_3, _ = model(img_2)
            barlow_loss_3 = barlow_criterion(x_ft_3, y_ft_3)
        scaler.scale(barlow_loss_3).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += barlow_loss_3

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            _, _, _, _, _, _, _, x_ft_4 = model(img_1)
            _, _, _, _, _, _, _, y_ft_4 = model(img_2)
            barlow_loss_4 = barlow_criterion(x_ft_4, y_ft_4) * 2
        scaler.scale(barlow_loss_4).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += barlow_loss_4
    print("train losses: ", losses)
    return losses

def train_fine(args, model, train_dl, optimizer, loss_fn, epoch):

    model.train()
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    losses = 0
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for img, _, labels in tqdm(train_dl, desc = "origin_fine_train", position = 1, leave = False):
        img, labels = img.to(args.device), labels.to(args.device)
        
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])

        r = np.random.rand(1)
        if r > args.cut_prob:
            pass
        else:
            lam = np.random.beta(args.beta2, args.beta2)
            rand_index = torch.randperm(img.size()[0]).to(args.device)
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
        
        table_11_img, table_22_img, table_33_img = list(), list(), list()
        for idx in range(img.size()[0]):
            table_1_img, table_2_img, table_3_img = jigsaw_generator(img[idx, :, :, :], args.patches[0]), jigsaw_generator(img[idx, :, :, :], args.patches[1]), jigsaw_generator(img[idx, :, :, :], args.patches[2])
            table_11_img.append(table_1_img)
            table_22_img.append(table_2_img)
            table_33_img.append(table_3_img)

        table_11_img, table_22_img, table_33_img = torch.stack(table_11_img, 0), torch.stack(table_22_img, 0), torch.stack(table_33_img, 0)
        table_11_img, table_22_img, table_33_img = table_11_img.to(args.device), table_22_img.to(args.device), table_33_img.to(args.device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            output_1, _, _, _, _, _, _, _ =  model(table_11_img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_1, labels)
            else:
                loss = loss_fn(output_1, target_a) * lam + loss_fn(output_1, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            _, output_2, _, _, _, _, _, _ = model(table_22_img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_2, labels)
            else:
                loss = loss_fn(output_2, target_a) * lam + loss_fn(output_2, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            _, _, output_3, _, _, _, _, _ = model(table_33_img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_3, labels)
            else:
                loss = loss_fn(output_3, target_a) * lam + loss_fn(output_3, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            _, _, _, output_4, _, _, _, _ = model(img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_4, labels) * 2 
            else:
                loss = (loss_fn(output_4, target_a) * lam + loss_fn(output_4, target_b) * (1. - lam)) * 2
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses
  
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

def val_fine(args, model, test_dl, loss_fn):
    model.eval()
    pred_list, label_list = torch.Tensor([]), torch.Tensor([])

    test_loss, total, correct = 0, 0, 0
    for idx, (image, _, label) in enumerate(tqdm(test_dl, desc = "Fine tester", position = 1, leave = False)):
        image, label = image.float().to(args.device), label.type(torch.LongTensor).to(args.device)

        with torch.cuda.amp.autocast():
            output_1, output_2, output_3, output_concat, _, _, _, _ = model(image)
            # output_1, output_2, output_3, output_concat = model(image)
            outputs_com = output_1 + output_2 + output_3 + output_concat
            loss = loss_fn(outputs_com, label)
        
        test_loss += loss.item()
        _, pred = torch.max(outputs_com.data, 1)
        total += label.size(0)
        correct += pred.eq(label.data).cpu().sum()

        pred_list = torch.cat([pred_list.cpu(), pred.cpu()], dim = 0)
        label_list = torch.cat([label_list.cpu(), label.cpu()], dim = 0)

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / len(test_dl)
    # test_f1, test_precision, test_recall = get_metrics(args, label_list, pred_list)

    # confusion matrix
    matrix = confusion_matrix(label_list, pred_list)
    acc_list = matrix.diagonal()/matrix.sum(axis=1)
    acc_list = [round(x*100, 2) for x in acc_list]

    class_name = ['00_Normal', '01_Spot', '03_Solid', '04_Dark Dust', '05_Shell', '06_Fiber', '07_Smear', '08_Pinhole', '11_OutFo']
    plot_confusion_matrix(matrix, class_name, './img/{}_confusion_matrix.png'.format('exp:6'))

    return {'loss': test_loss,
            'total_acc': test_acc,
            'class_acc': acc_list}

def val_dino(args, val_loader, model, criterion):

    model.eval()
    n_image = len(val_loader.dataset)
    n_batch = len(val_loader)
    bs = args.batch_size
    total_loss = 0.

    total_preds_per_classifier = {k: torch.zeros(n_image, dtype=torch.long) for k in model.linear_classifiers.classifiers_dict.keys()}
    total_labels = torch.zeros(n_image, dtype=torch.long)

    with torch.no_grad():
        for batch_idx, (images, _, labels) in enumerate(val_loader):
            labels = labels.cuda()

            logits = model(images)
            losses = {f"loss_{k}": criterion(v, labels) for k, v in logits.items()}
            loss = sum(losses.values())
            total_loss += loss.item()
            total_idx = bs * batch_idx

            # Process outputs for each classifier
            for k, v in logits.items():
                preds = torch.argmax(v, dim=1)
        
                if batch_idx == n_batch - 1:
                    total_preds_per_classifier[k][total_idx:] = preds.cpu()
                    total_labels[total_idx:] = labels.cpu()
                else:
                    total_preds_per_classifier[k][total_idx:total_idx + bs] = preds.cpu()
                    total_labels[total_idx:total_idx + bs] = labels.cpu()

    # Compute accuracy for each classifier
    results = {}
    for k, preds in total_preds_per_classifier.items():
        total_acc = (total_labels == preds).sum().item() / n_image * 100
        total_loss /= n_image
        matrix = confusion_matrix(total_labels.numpy(), preds.numpy())
        acc_list = matrix.diagonal() / matrix.sum(axis=1)
        acc_list = [round(x * 100, 2) for x in acc_list]
        
        # Save results 
        results[k] = {
            'loss' : total_loss,
            'total_acc': total_acc,
            'class_acc': acc_list,
            'confusion_matrix': matrix
        }

        # plot confusion
        # class_name = ['00_Normal', '01_Spot', '03_Solid', '04_Dark Dust', '05_Shell', '06_Fiber', '07_Smear', '08_Pinhole', '11_OutFo']
        # plot_confusion_matrix(matrix, class_name, './img/{}_confusion_matrix_ACC_{:.3f}.png'.format(k, total_acc))

    best_key = max(results, key=lambda k: results[k]['total_acc'])
    return results[best_key], best_key


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    torch.autograd.set_detect_anomaly(True)
    args = get_opts()
    print(json.dumps(vars(args), indent='\t'), flush=True)
    main(args)