import numpy as np
import pandas as pd
import torch
import h5py
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split
from util import box_ops
import random


def toTensor(img, **params):
    return F.to_tensor(img)


def load_and_clean_data(path):
    """
    Takes care of loading, cleaning and the splitting the data into the train and test sets.
    :param path: input H5 data object path
    :return: cleaned train and test sets containing img, boxes and labels.
    """
    data = h5py.File(path, 'r')
    cleaned_data = []
    for idx in range(len(data['rdms'])):
        cleaned_frame = []
        # check to see if at least one object of interest exists in the frame.
        if len(data['labels'][str(idx)]) is not 0:
            img = np.array(data['rdms'][idx])
            h, w = img.shape[0], img.shape[1]
            # add the objects' boxes and labels into new lists
            boxes = []
            labels = []
            for j in range(len(data['labels'][str(idx)])):
                # check to see if the label is not "0" ('no object') and proceed
                if data['labels'][str(idx)][j][-1] is not 3:
                    xmin = data['labels'][str(idx)][j][0] / w
                    ymin = data['labels'][str(idx)][j][1] / h
                    xmax = data['labels'][str(idx)][j][2] / w
                    ymax = data['labels'][str(idx)][j][3] / h
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(data['labels'][str(idx)][j][-1])
            # if length of the box is not zero, then add the obj of interest to the frame list.
            if len(boxes) is not 0:
                cleaned_frame.append(img)
                cleaned_frame.append(boxes)
                cleaned_frame.append(labels)
                assert len(boxes) == len(labels)
        # if the frame contains at least one obj, then add it to the final data list.
        if len(cleaned_frame) is not 0:
            cleaned_data.append(cleaned_frame)
    # split the dataset into train and test
    data_df = pd.DataFrame(cleaned_data, columns=["img", "boxes", "labels"])

    train, test = train_test_split(data_df, test_size=0.2, random_state=42, shuffle=True)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    return train, test


class CustomDataset(Dataset):

    def __init__(self, data_df, split='train'):

        self.split = split
        self.data = data_df
        special_aug_list = [A.HorizontalFlip(p=1), A.Flip(p=1),  # vertical
                            A.Transpose(p=1), A.RandomRotate90(p=1), A.RandomSizedBBoxSafeCrop(256, 32, p=1),
                            A.LongestMaxSize(p=1), ]
        random_special_aug = random.choices([A.NoOp(), random.choice(special_aug_list)], weights=[0.4, 0.6])[0]

        if split == 'train':
            self.transform = A.Compose(
                [random_special_aug, A.Normalize([385], [2691.76]), A.Resize(300, 300), A.Lambda(p=1, image=toTensor)],
                bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        if split == 'test':
            self.transform = A.Compose([A.Normalize([385], [2691.76]), A.Resize(300, 300), A.Lambda(p=1, image=toTensor)],
                                       bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    def __getitem__(self, idx):
        """
        Load the object image and its target for the given index.
        :param idx: index of the radar frame to be read
        :return: radar_frame and its corresponding target list
        """
        # load the image
        print(idx)
        img = np.array(self.data.iloc[idx]['img'])
        h, w = img.shape[0], img.shape[1]
        # convert boxes, labels and image id into a torch.Tensor
        boxes = torch.tensor(self.data.iloc[idx]["boxes"], dtype=torch.float32)
        labels = torch.tensor(self.data.iloc[idx]["labels"], dtype=torch.float32)
        img_id = torch.Tensor(idx)

        area = h * w * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros(len(self.data.iloc[idx]['labels']), dtype=torch.int64)

        transformed = self.transform(image=img, bboxes=boxes, category_ids=labels)
        img = transformed["image"]
        
        target = {"boxes": torch.tensor(transformed["bboxes"], dtype=torch.float32), "labels": labels,}
                  # "image_id": img_id, "area": area, "iscrowd": iscrowd}

        for i, bboxes in enumerate(target['boxes']):
            target['boxes'][i] = box_ops.box_xyxy_to_cxcywh(bboxes)

        return img.repeat(3, 1, 1), torch.tensor((target['boxes']),dtype=torch.float32)[None,...], torch.tensor((target['labels']))[None,...], torch.zeros(1)

    def __len__(self):
        if self.split == 'train':
            return int(len(self.data) * 0.8)
        else:
            return int(len(self.data) * 0.2)


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    boxes = list()
    labels = list()
    difficulties = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        difficulties.append(b[3])

    for i in range(len(boxes)):
        boxes[i] = boxes[i].squeeze(0)
        labels[i] = labels[i].squeeze(0)
    images = torch.stack(images, dim=0)

    return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

"""
to run:

train, test = load_and_clean_data(path)

train_dataset = CustomDataset(train, split="train")
test_dataset = CustomDataset(test, split="test")

train_dataloader = DataLoader(train_dataset)

"""
# train, test = load_and_clean_data('../../data/data.h5')
# dataset = CustomDataset(train, split='train')
# data_loader = DataLoader(dataset)
#
# for img, box, label, _ in data_loader:
#     print()

