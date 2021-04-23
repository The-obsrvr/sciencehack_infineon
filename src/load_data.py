import os
import numpy as np
import torch
import h5py


class DataGenerator(object):

    def __init__(self, path):

        self.data = h5py.File(path, 'r')
        
    def __getitem__(self, idx):
        """
        Load the object image and its target for the given index.

        :param idx: index of the radar frame to be read
        :return: radar_frame and its corresponding target list
        """

        # load the image
        img = np.array(self.data['rdms'][idx])

        # add the objects' boxes and labels into new lists
        boxes = []
        labels = []
        if len(self.data['labels'][str(idx)]) is not 0:
            for j in range(len(self.data['labels'][str(idx)])):
                xmin = self.data['labels'][str(idx)][j][0]
                ymin = self.data['labels'][str(idx)][j][1]
                xmax = self.data['labels'][str(idx)][j][2]
                ymax = self.data['labels'][str(idx)][j][3]
                boxes.append([xmin, ymin, xmax, ymax])

                labels.append(self.data['labels'][str(idx)][j][-1])
        else:
            boxes.append(None)
            labels.append(None)

        # convert box into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # convert label into a torch.Tensor
        labels = torch.as_tensor(labels, dtype=torch.float32)

        img_id = torch.Tensor(idx)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros(len(self.data['labels'][str(idx)]), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.imgs)


