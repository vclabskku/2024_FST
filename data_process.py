import json
import os
import random

data_root = '../data/251212_G2_Rev_Accumulate_v3'
label_list = ['00_Normal', '01_Spot', '03_Solid', '04_Dark Dust', '05_Shell', '06_Fiber', '07_Smear', '08_Pinhole', '11_OutFo']

# [266,186,121,25,467,2,377,130,102  ]

label_dict = {}
for i, label in enumerate(label_list):
    label_dict[label] = i

train_data = {}
val_data = {}
test_data = {}
for label_name in os.listdir(data_root):
    if label_name not in label_list:
        continue
    label_path = os.path.join(data_root, label_name)

    image_paths = []
    if os.path.isdir(label_path):
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            if os.path.isfile(image_path):
                image_paths.append(image_path)

    random.shuffle(image_paths)
    split_point_test = int(len(image_paths) * 0.8)

    train_image_paths = image_paths[:split_point_test]
    test_image_paths = image_paths[split_point_test:]
    print(f"{label_name} - train: {len(train_image_paths)}, test: {len(test_image_paths)}")

    label = label_dict[label_name]
    for mode, image_paths in zip(['train', 'test'], [train_image_paths, test_image_paths]):
        for path in image_paths:
            if mode == 'train':
                data_key = len(train_data)
                train_data[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}
            elif mode == 'test':
                data_key = len(test_data)
                test_data[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}

with open(os.path.join(data_root, 'train.json'), 'w') as json_file:
    json.dump(train_data, json_file, indent=4)

with open(os.path.join(data_root, 'test.json'), 'w') as json_file:
    json.dump(test_data, json_file, indent=4)