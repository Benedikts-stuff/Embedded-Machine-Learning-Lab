import torch
from torchinfo import summary
import os, json
import numpy as np
from PIL import Image

import torchvision
from torchvision import transforms as tf
from Training.data_augmentation import PersonAugmentation

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

class VOCTransform:
    def __init__(self, train=True, only_person=False, is_baseline=False):
        self.only_person = only_person
        self.train = train
        self.is_baseline = is_baseline
        if train:
            self.augmentation = PersonAugmentation(is_train=True, is_baseline=False)
        else:
            self.augmentation = PersonAugmentation(is_train=False, is_baseline=is_baseline)

    def __call__(self, image, target):
        num_bboxes = 10
        width, height = 320, 320 # YOLO Input Size

        image_np = np.array(image)
        
        target_objs = target['annotation']['object']
        raw_bboxes = []
        raw_labels = []

        for item in target_objs:
            xmin = float(item['bndbox']['xmin'])
            ymin = float(item['bndbox']['ymin'])
            xmax = float(item['bndbox']['xmax'])
            ymax = float(item['bndbox']['ymax'])
            
            label = class_to_num(item['name'])
            
            if self.only_person:
                if label == class_to_num("person"):
                    raw_bboxes.append([xmin, ymin, xmax, ymax])
                    raw_labels.append(0) # 0 für Person
            else:
                raw_bboxes.append([xmin, ymin, xmax, ymax])
                raw_labels.append(label)

        augmented = self.augmentation(
            image=image_np, 
            bboxes=raw_bboxes, 
            class_labels=raw_labels
        )
        
        image_tensor = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        if self.is_baseline and image_tensor.max() > 1.0:
            image_tensor = image_tensor.float() / 255.0

        target_vectors = []
        for bbox, lbl in zip(aug_bboxes, aug_labels):
            xmin, ymin, xmax, ymax = bbox
            
            w = (xmax - xmin)
            h = (ymax - ymin)
            cx = xmin + w/2
            cy = ymin + h/2
            
            target_vector = [
                cx / width,
                cy / height,
                w / width,
                h / height,
                1.0, # Objectness
                float(lbl)
            ]
            target_vectors.append(target_vector)

        if len(target_vectors) == 0:
            target_vectors = torch.zeros((num_bboxes, 6))
            target_vectors[:, -1] = -1
        else:
            target_vectors = list(sorted(target_vectors, key=lambda x: x[2]*x[3]))
            target_vectors = torch.tensor(target_vectors)
            
            if target_vectors.shape[0] < num_bboxes:
                zeros = torch.zeros((num_bboxes - target_vectors.shape[0], 6))
                zeros[:, -1] = -1
                target_vectors = torch.cat([target_vectors, zeros], 0)
            elif target_vectors.shape[0] > num_bboxes:
                target_vectors = target_vectors[:num_bboxes]

        return image_tensor, target_vectors


def VOCDataLoader(train=True, batch_size=32, shuffle=False, is_baseline=False):
    image_set = "train" if train else "val"
    
    dataset = torchvision.datasets.VOCDetection(
        "data/", year="2012", image_set=image_set, download=False, 
        transforms=VOCTransform(train=train, is_baseline=is_baseline)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
def VOCDataLoaderPerson(train=True, batch_size=32, shuffle=False, is_baseline=False):
    image_set = "train" if train else "val"
    
    dataset = torchvision.datasets.VOCDetection(
        "data/", year="2012", image_set=image_set, download=False,
        transforms=VOCTransform(train=train, only_person=True, is_baseline=is_baseline)
    )
    with open("data/person_indices.json", "r") as fd: 
        indices = list(json.load(fd)[image_set])
    dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)