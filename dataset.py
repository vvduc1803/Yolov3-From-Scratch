from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import xml.etree.ElementTree as ET
import config
import numpy as np
import torch
from utils import iou_width_height as iou
import matplotlib.pyplot as plt

class Pascal_Data(Dataset):
    def __init__(self,
                 img_root,
                 annotation_root,
                 anchors=config.ANCHORS,
                 class_names=config.PASCAL_CLASSES,
                 image_size=416,
                 S=[13, 26, 52],
                 transform=None):

        self.img_root = img_root
        self.annotation_root = annotation_root
        self.annotation_files = os.listdir(annotation_root)
        self.class_names = class_names
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotation_files)
    def __getitem__(self, idx):
        print(len(self.annotation_files))
        annotation_path = os.path.join(self.annotation_root, self.annotation_files[idx])

        tree = ET.parse(annotation_path)
        root = tree.getroot()
        objects = [x for x in root if 'object' in str(x)]
        bboxes = []
        # for object in objects:
        #     class_name = self.class_names.index(object[0].text)
        #     x_min, y_min, x_max, y_max = int(object[4][0].text), int(object[4][1].text), int(object[4][2].text), int(
        #         object[4][3].text)
        #     bboxes.append([x_min, y_min, x_max, y_max, class_name])
        # bboxes = np.array(bboxes)

        for elem in tree.iter():
            if elem.tag == 'width':
                width = int(elem.text)
            if elem.tag == 'height':
                height = int(elem.text)
            if elem.tag == 'filename':
                img_file = elem.text
            if elem.tag == 'object':
                box = []
                for attr in list(elem):
                    if attr.tag == 'name':
                        box.append(self.class_names.index(attr.text))
                    if attr.tag == 'bndbox':
                        for dim in list(attr):
                            box.append(int(dim.text))
                bboxes.append(box)
        bboxes = np.roll(np.array(bboxes), 4, axis=1).tolist()

        img_path = os.path.join(self.img_root, img_file)
        img = np.array(Image.open(img_path).convert('RGB'))/255.

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            x_min, y_min, x_max, y_max, class_label = box
            x_min, y_min, x_max, y_max = x_min / width, y_min / height, x_max / width, y_max / height
            x, y, width, height = (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min

            iou_anchors = iou(torch.tensor([width, height]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                S = self.S[scale_idx]
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
                    print(scale_idx, anchor_on_scale, i, j)
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return img, tuple(targets)

def test():
    from utils import plot_image, cells_to_bboxes
    from utils import non_max_suppression as nms
    dataset = Pascal_Data('data/train_val/JPEGImages', 'data/train_val/Annotations')
    img, y = dataset.__getitem__(100)
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(config.ANCHORS) / (
            1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []
        print(x.shape)

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



