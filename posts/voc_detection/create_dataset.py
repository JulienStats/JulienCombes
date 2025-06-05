from torchvision.datasets import VOCDetection
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import draw_bounding_boxes
from typing import Dict, Optional
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

categories = ['pottedplant', 'bottle', 'chair', 'diningtable', 'person', 'car', 'train', 'bus', 'bicycle', 'cat', 'aeroplane', 'tvmonitor', 'sofa', 'sheep', 'dog', 'bird', 'motorbike', 'horse', 'boat', 'cow']


def unpack_box(box_dict:Dict):
    """
    Unpack the box dictionary into a list of coordinates
    """
    return torch.from_numpy(np.array([
        box_dict["xmin"],
        box_dict["ymin"],
        box_dict["xmax"],
        box_dict["ymax"]
    ], dtype = float))

def annotation_to_torch(target:Dict):
    rep = {}
    detections = target["annotation"]["object"]
    
    rep["labels"] = np.array([categories.index(i["name"]) for i in detections])
    # xmin ymin xmax ymax
    rep["boxes"] = torch.stack([unpack_box(i["bndbox"]) for i in detections])

    return rep    

class MyVoc(VOCDetection):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        # Apply your transformations here
        target = annotation_to_torch(target)

        transform = A.Compose([
            A.PadIfNeeded(500,500),
            A.RandomCrop(400,400),
            A.Resize(224, 224),
            A.Normalize(max_pixel_value=255),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc', # Specify input format
            label_fields=['class_labels'],
            filter_invalid_bboxes=True)
        )

        transformed = transform(
            image=np.array(image), 
            bboxes=target["boxes"], 
            class_labels=target["labels"]
        )

        transformed["labels"] = transformed["class_labels"]
        transformed["boxes"] = transformed["bboxes"]
        transformed.pop("class_labels")
        transformed.pop("bboxes")
        return transformed

    def draw_item(self, index:Optional[int] = None):
        if index is None:
            index = np.random.randint(0, len(self))
            print(index)

        item = self[index]
        image = item.pop("image")

        labels = item
        print(labels)

        with_boxes = draw_bounding_boxes(
            image =  image,
            boxes=torch.from_numpy(labels["boxes"]),
            labels = [categories[i] for i in labels["labels"]],
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(with_boxes.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.show()
        return with_boxes


def create_dataset(data_root:str):
    ds = MyVoc(
        root = data_root, 
        download = False
    )
    
    return ds

if __name__ == "__main__":
    ds = create_dataset("~/data_wsl/voc")

    ds.draw_item()
