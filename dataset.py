import os
import glob
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from utils.utils import extracting_bboxes
class SolarPannelsDataset(torch.utils.data.Dataset):
    def __init__(self, transforms):
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.img_paths = sorted(glob.glob("data/solar_pannels/*.jpg"))
        self.bboxes_paths = sorted(glob.glob("data/solar_pannels/*.txt"))

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.img_paths[idx]
        bboxes_path = self.bboxes_paths[idx]
        img = read_image(img_path)
        bboxes = extracting_bboxes(img_path, bboxes_path)
        num_objs = len(bboxes) # number of objects in the image
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=F.get_size(img))
        # target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = idx
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)