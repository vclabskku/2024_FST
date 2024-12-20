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

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.args.image_cut == 'crop':
            size_transform = transforms.CenterCrop(args.resolution)
        elif self.args.image_cut == 'resize':
            size_transform = transforms.Resize((args.resolution, args.resolution))

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
            img = self.transform(img)
        return img, names, target

    def __len__(self) -> int:
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