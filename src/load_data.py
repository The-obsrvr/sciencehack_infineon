import os
import numpy as np
import torch
import h5py
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from util import box_ops
import random

def toTensor(img, **params):
    return F.to_tensor(img)


class CustomDataset(Dataset):

    def __init__(self, path, split='train'):

        self.data = h5py.File(path, 'r')
        self.split = split

        spacial_aug_list = [A.HorizontalFlip(p=1), A.Flip(p=1),  # vertical
                            A.Transpose(p=1), A.RandomRotate90(p=1), A.RandomSizedBBoxSafeCrop(256, 32, p=1),
                            A.LongestMaxSize(p=1), ]
        random_spacial_aug = random.choices([A.NoOp(), random.choice(spacial_aug_list)], weights=[0.4, 0.6])[0]

        if split == 'train':
            self.transform = A.Compose([random_spacial_aug, A.Normalize([385], [2691.76]), A.Lambda(p=1, image=toTensor)],
                                            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        if split == 'test':
            self.transform = A.Compose([A.Normalize([385], [2691.76]), A.Lambda(p=1, image=toTensor)], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    def __getitem__(self, idx):
        """
        Load the object image and its target for the given index.

        :param idx: index of the radar frame to be read
        :return: radar_frame and its corresponding target list
        """

        # load the image
        img = np.array(self.data['rdms'][idx])
        h ,w = img.shape[0], img.shape[1]
        # add the objects' boxes and labels into new lists
        boxes = []
        labels = []
        if len(self.data['labels'][str(idx)]) is not 0:
            for j in range(len(self.data['labels'][str(idx)])):
                xmin = self.data['labels'][str(idx)][j][0] / w
                ymin = self.data['labels'][str(idx)][j][1] / h
                xmax = self.data['labels'][str(idx)][j][2] / w
                ymax = self.data['labels'][str(idx)][j][3] / h
                boxes.append([xmin, ymin, xmax, ymax])

                labels.append(self.data['labels'][str(idx)][j][-1])
        else:
            boxes.append(None)
            labels.append(None)

        # convert box into a torch.Tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        # convert label into a torch.Tensor
        labels = torch.tensor(labels, dtype=torch.float32)

        img_id = torch.Tensor(idx)
        area = h * w *(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros(len(self.data['labels'][str(idx)]), dtype=torch.int64)

        target = {}
        # target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        transformed = self.transform(image=img, bboxes=boxes, category_ids=labels)
        img = transformed["image"]
        target["boxes"] = torch.tensor(transformed["bboxes"])

        for idx, bboxes in enumerate(target['boxes']):
            target['boxes'][idx] = box_ops.box_xyxy_to_cxcywh(bboxes)

        return img, target

    def __len__(self):
        return self.data['rdms'].shape[0]

dataset = CustomDataset('../data/data.h5', split='test')
data_loader = DataLoader(dataset)

for img, target in data_loader:
    print()



