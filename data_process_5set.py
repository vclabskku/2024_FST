import json
import os
import random

data_root = '../data/251212_G2_Rev_Accumulate_v2'
label_list = ['00_Normal', '01_Spot', '03_Solid', '04_Dark Dust', '05_Shell', '06_Fiber', '07_Smear', '08_Pinhole', '10_Party', '11_OutFo']

# [266,186,121,25,467,2,377,130,102  ]

label_dict = {}
for i, label in enumerate(label_list):
    label_dict[label] = i

train_data_0 = {}
test_data_0 = {}
train_data_1 = {}
test_data_1 = {}
train_data_2 = {}
test_data_2 = {}
train_data_3 = {}
test_data_3 = {}
train_data_4 = {}
test_data_4 = {}
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
    if len(image_paths) < 5:
        lip = len(image_paths)
        train_image_paths_list = [image_paths[:lip - 1],
                                  image_paths[:lip - 1],
                                  image_paths[:lip - 2] + image_paths[lip-1:],
                                  image_paths[:lip - 2] + image_paths[lip - 1:],
                                  image_paths[1:]
                                  ]

        test_image_paths_list = [image_paths[-1:],
                                 image_paths[-1:],
                                 image_paths[:lip - 2:lip - 1],
                                 image_paths[:lip - 2:lip - 1],
                                 image_paths[:1]
                                 ]
    else:
        split_point_1 = int(len(image_paths) * 0.2)
        split_point_2 = int(len(image_paths) * 0.4)
        split_point_3 = int(len(image_paths) * 0.6)
        split_point_4 = int(len(image_paths) * 0.8)

        train_image_paths_list = [image_paths[:split_point_4],
                             image_paths[:split_point_3] + image_paths[split_point_4:],
                             image_paths[:split_point_2] + image_paths[split_point_3:],
                             image_paths[:split_point_1] + image_paths[split_point_2:],
                             image_paths[split_point_1:],
                             ]

        test_image_paths_list = [image_paths[split_point_4:],
                            image_paths[split_point_3:split_point_4],
                            image_paths[split_point_2:split_point_3],
                            image_paths[split_point_1:split_point_2],
                            image_paths[:split_point_1]
                            ]
    for i in range(len(train_image_paths_list)):

        train_image_paths = train_image_paths_list[i]
        test_image_paths = test_image_paths_list[i]
        print(f"{label_name} - train: {len(train_image_paths)}, test: {len(test_image_paths)}")

        label = label_dict[label_name]
        for mode, image_paths in zip(['train', 'test'], [train_image_paths, test_image_paths]):
            for path in image_paths:
                if i == 0:
                    if mode == 'train':
                        data_key = len(train_data_0)
                        train_data_0[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}
                    elif mode == 'test':
                        data_key = len(test_data_0)
                        test_data_0[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}
                elif i == 1:
                    if mode == 'train':
                        data_key = len(train_data_1)
                        train_data_1[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}
                    elif mode == 'test':
                        data_key = len(test_data_1)
                        test_data_1[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}
                elif i == 2:
                    if mode == 'train':
                        data_key = len(train_data_2)
                        train_data_2[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}
                    elif mode == 'test':
                        data_key = len(test_data_2)
                        test_data_2[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}
                elif i == 3:
                    if mode == 'train':
                        data_key = len(train_data_3)
                        train_data_3[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}
                    elif mode == 'test':
                        data_key = len(test_data_3)
                        test_data_3[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}
                elif i == 4:
                    if mode == 'train':
                        data_key = len(train_data_4)
                        train_data_4[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}
                    elif mode == 'test':
                        data_key = len(test_data_4)
                        test_data_4[f"{data_key}"] = {'image_path': path, 'label': label, 'label_name': label_name, 'mode': mode}

with open(os.path.join(data_root, f'train_0.json'), 'w') as json_file:
    json.dump(train_data_0, json_file, indent=4)

with open(os.path.join(data_root, f'test_0.json'), 'w') as json_file:
    json.dump(test_data_0, json_file, indent=4)

with open(os.path.join(data_root, f'train_1.json'), 'w') as json_file:
    json.dump(train_data_1, json_file, indent=4)

with open(os.path.join(data_root, f'test_1.json'), 'w') as json_file:
    json.dump(test_data_1, json_file, indent=4)

with open(os.path.join(data_root, f'train_2.json'), 'w') as json_file:
    json.dump(train_data_2, json_file, indent=4)

with open(os.path.join(data_root, f'test_2.json'), 'w') as json_file:
    json.dump(test_data_2, json_file, indent=4)

with open(os.path.join(data_root, f'train_3.json'), 'w') as json_file:
    json.dump(train_data_3, json_file, indent=4)

with open(os.path.join(data_root, f'test_3.json'), 'w') as json_file:
    json.dump(test_data_3, json_file, indent=4)

with open(os.path.join(data_root, f'train_4.json'), 'w') as json_file:
    json.dump(train_data_4, json_file, indent=4)

with open(os.path.join(data_root, f'test_4.json'), 'w') as json_file:
    json.dump(test_data_4, json_file, indent=4)