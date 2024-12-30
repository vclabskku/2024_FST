import os
import json

import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F
import torchvision.transforms as transforms
from dataset.utils import *
# from .utils import ColorJitter, Lighting
from utils.opt import get_opts
from utils.fgssl import jigsaw_generator

class FSTDataset(Dataset):
    def __init__(self, args, data_json, is_train=True, augment=['flip', 'rotate'], crop=True, model_type='imagenet'):
        self.args = args
        with open(data_json, 'r') as json_file:
            data = json.load(json_file)
        images = []
        labels = []
        image_names = []
        '''
        label list 
        ['00_Normal', '01_Spot', '03_Solid', '04_Dark Dust', '05_Shell', '06_Fiber', '07_Smear', '08_Pinhole', '11_OutFo']
        '''
        for key, value in data.items():
            image = Image.open(value['image_path']).convert('RGB')
            images.append(image)
            labels.append(value['label'])
            image_names.append(value['image_path'])
        print('class list : ', set(labels))
        class_list = list(set(labels))
        self.class_list = class_list
        class_map = {}
        adjusted_label = 0
        self.num_classes = len(class_list)
        # 존재하는 클래스에만 라벨을 매핑. 현재 3번 클래스 데이터 없어서 아래 과정 추가.
        for cls in class_list:
            if cls in labels:
                class_map[cls] = adjusted_label
                adjusted_label += 1
        adjusted_labels = [class_map[label] for label in labels]

        self.images = images
        self.labels = adjusted_labels
        self.image_names = image_names

        self.cls_num_list = []
        for i in set(self.labels):
            self.cls_num_list.append(self.labels.count(i))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.args.image_cut == 'crop':
            size_transform = transforms.CenterCrop(args.resolution)
        elif self.args.image_cut == 'resize':
            size_transform = transforms.Resize((args.resolution, args.resolution))
        elif self.args.image_cut == 'patch':
            patch_size = int(args.patch_num**0.5)
            size_transform = transforms.Resize((args.resolution*patch_size, args.resolution*patch_size))


        if is_train:
            self.transform = transforms.Compose([
                size_transform,
                transforms.RandomRotation(180) if 'rotate' in augment else transforms.Lambda(lambda x: x),
                transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(p=0.5) if args.h_flip else transforms.RandomHorizontalFlip(p=0.0),
                    transforms.RandomVerticalFlip(p=0.5) if args.v_flip else transforms.RandomVerticalFlip(p=0.0),
                ]) if 'flip' in augment else transforms.Lambda(lambda x: x),

                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                size_transform,
                transforms.ToTensor(),
                normalize,
            ])
        
        if self.args.concat:
            self.crop_transform = transforms.Compose([
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize
            ])

            self.resize_transform = transforms.Compose([
                transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                normalize
            ])

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, names = self.images[index], self.labels[index], self.image_names[index]
        if self.transform is not None:
            if self.resize_transform is not None and self.crop_transform is not None: #self.args.concat == True
                img1 = self.resize_transform(img)
                img2 = self.crop_transform(img)
                img = [img1, img2]
            else:
                img = self.transform(img)
        return img, names, target

    def __len__(self) -> int:
        return len(self.images)

class barlowDataset(Dataset):
    def __init__(self, args, data_json):
        self.args = args
        with open(data_json, 'r') as json_file:
            data = json.load(json_file)
        images = []
        labels = []
        for key, value in data.items():
            images.append(value['image_path'])
            labels.append(value['label'])
        print('class list : ', set(labels))

        self.images = images
        self.labels = labels
        self.transforms_1 = transforms.Compose([
                    transforms.CenterCrop(args.resolution),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

        self.transforms_2 = transforms.Compose([
                    transforms.CenterCrop(args.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])


    def __getitem__(self, idx):
        image_name = self.images[idx]
        img = Image.open(image_name).convert('RGB')
        labels = self.labels[idx]

        img_1 = self.transforms_1(img)   # origin
        img_2 = self.transforms_2(img)   # transform
        table_1_img, table_2_img, table_3_img = jigsaw_generator(img_1, self.args.patches[0]), jigsaw_generator(img_1, self.args.patches[1]), jigsaw_generator(img_1, self.args.patches[2])
            # 2         # 4         # 8
        return  img_1, table_1_img,  table_2_img, table_3_img, img_2

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    print(">> Load the data...")
    train_dataset = FSTDataset('./data/train_0.json', is_train=True, augment=['flip'], crop=True, model_type='cifar')
    val_dataset = FSTDataset('./data/test_0.json', is_train=False, augment=['flip'], crop=True, model_type='cifar')
    print("Train: {:d} / Val: {:d}".format(len(train_dataset), len(val_dataset)))

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=1,
        shuffle=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=1,
        shuffle=False,
        drop_last=False)

    print("<< Finished.\n")