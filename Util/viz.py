import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np

from typing import List

CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    )


def class_to_num(class_str):
    for idx, string in enumerate(CLASSES):
        if string == class_str: return idx

def num_to_class(number):
    for idx, string in enumerate(CLASSES):
        if idx == number: return string
    return 'none'


def display_result(image: torch.Tensor, output: List[torch.tensor], target: torch.Tensor, file_path='image.png') -> None:
    _, ax = plt.subplots()

    pad = 20
    image = image.numpy()[0,:].transpose(1,2,0)
    image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    ax.imshow(image)
    
    img_shape = 320
    if output:
        bboxes = torch.stack(output, dim=0)
        for i in range(bboxes.shape[1]):

            if bboxes[0,i,-1] >= 0:
                cx = int(bboxes[0,i,0]*img_shape - bboxes[0,i,2]*img_shape/2) + pad
                cy = int(bboxes[0,i,1]*img_shape - bboxes[0,i,3]*img_shape/2) + pad

                w = int(bboxes[0,i,2]*img_shape)
                h = int(bboxes[0,i,3]*img_shape)

                rect = patches.Rectangle((cx,cy),
                                        w, h, linewidth=2, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
                ax.annotate(num_to_class(int(bboxes[0,i,5])) + " "+  f"{float(bboxes[0,i,4]):.2f}",(cx,cy), color='r')

    for i in range(target.shape[1]):
        if target[0,i,-1] >= 0:
            cx = int(target[0,i,0]*img_shape - target[0,i,2]*img_shape/2) + pad
            cy = int(target[0,i,1]*img_shape - target[0,i,3]*img_shape/2) + pad

            w = target[0,i,2]*img_shape
            h = target[0,i,3]*img_shape

            rect = patches.Rectangle((cx,cy),
                                    w, h, linewidth=2, facecolor='none', edgecolor='g')
            ax.add_patch(rect)
            ax.annotate(num_to_class(int(target[0,i,5])),(cx,cy), color='g')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()
    plt.close()