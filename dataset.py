# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import xml.etree.ElementTree as ET
import config
import numpy as np
import torch
from utils import iou_width_height as iou

class Pascal_Data(Dataset):
    """
    Build custom dataset for pascal-voc dataset
    Args:
        img_root: root of image file (ex: data/train_val/JPEGImages)
        annotation_root: root of annotation file (ex: data/train_val/Annotations)
        anchors: 9 anchors use in object detection for image size (416x416)
        class_names: name of classes (ex: car, dog, cat, ...)
        S: 3 size of output targets in this case are 13x13, 26x26, 52x52
        transform: transform for image and bbox

    Returns:
        img: image after transform
        targets: tuples of 3 head after process store information of bbox, class and confident score
    """
    def __init__(self,
                 img_root,
                 annotation_root,
                 anchors=config.ANCHORS,
                 class_names=config.PASCAL_CLASSES,
                 S=[13, 26, 52],
                 transform=None):

        self.img_root = img_root
        self.annotation_root = annotation_root
        self.annotation_files = os.listdir(annotation_root)
        self.class_names = class_names
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):

        # Load annotation path
        annotation_path = os.path.join(self.annotation_root, self.annotation_files[idx])
        tree = ET.parse(annotation_path)

        # Initialize list store bbox and label
        bboxes = []

        # Process annotation file (format: .xml)
        for elem in tree.iter():

            # Take width and height of original image
            if elem.tag == 'width':
                ori_width = int(elem.text)
            if elem.tag == 'height':
                ori_height = int(elem.text)

            # Take name of image file
            if elem.tag == 'filename':
                img_file = elem.text

            # Take bounding box and label
            if elem.tag == 'object':
                box = []
                for attr in list(elem):
                    if attr.tag == 'name':
                        box.append(self.class_names.index(attr.text))
                    if attr.tag == 'bndbox':
                        for dim in list(attr):
                            if dim.tag == 'xmin':
                                xmin = int(round(float(dim.text)))
                            if dim.tag == 'ymin':
                                ymin = int(round(float(dim.text)))
                            if dim.tag == 'xmax':
                                xmax = int(round(float(dim.text)))
                            if dim.tag == 'ymax':
                                ymax = int(round(float(dim.text)))
                        box.extend([xmin, ymin, xmax, ymax])
                bboxes.append(box)

        # Roll label to the end
        bboxes = np.roll(np.array(bboxes), 4, axis=1).tolist()

        # Load image
        img_path = os.path.join(self.img_root, img_file)
        img = np.array(Image.open(img_path).convert('RGB'))

        # Apply transform for image and bboxes list
        if self.transform:
            augmentations = self.transform(image=img, bboxes=bboxes)
            img = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Initialize 3 head
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        # Load over all bounding box
        for box in bboxes:

            # Take information of box
            x_min, y_min, x_max, y_max, class_label = box

            # Normalize information
            x_min, y_min, x_max, y_max = x_min / ori_width, y_min / ori_height, x_max / ori_width, y_max / ori_height

            # Covert corners form to midpoint form
            x, y, width, height = (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min

            # Compute iou with each anchors to choose scale
            iou_anchors = iou(torch.tensor([width, height]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            has_anchor = [False] * 3  # each scale should have three anchor

            # Load over indices
            for anchor_idx in anchor_indices:

                # Choose scale
                scale_idx = anchor_idx // self.num_anchors_per_scale
                S = self.S[scale_idx]

                # Choose anchor in this scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                # Choose cell in this scale anh this anchor
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return img, tuple(targets)

def test():
    from utils import plot_image, cells_to_bboxes
    from utils import non_max_suppression as nms

    dataset = Pascal_Data('data/train_val/JPEGImages', 'data/train_val/Annotations')
    _, y = dataset.__getitem__(100)

    S = [13, 26, 52]

    scaled_anchors = torch.tensor(config.ANCHORS) / (
            1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]

            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]

            break
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")

        plot_image(x[0].to("cpu"), boxes)


if __name__ == '__main__':
    test()



