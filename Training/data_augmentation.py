import albumentations as A
from albumentations.pytorch import ToTensorV2

class PersonAugmentation:
    def __init__(self, is_train=True, is_baseline=False):

        if is_baseline:
            self.transform = A.Compose([
                A.Resize(320, 320),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        elif is_train:
            self.transform = A.Compose([
                A.Resize(320,320),
                A.HorizontalFlip(p=0.5),
                A.RandomSizedBBoxSafeCrop(width=320, height=320, p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, p=0.5),
                ], p=0.3),
                A.GaussNoise(std_range=(0.1, 0.3), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
        else:
            self.transform = A.Compose([
                A.Resize(320,320),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))

    def __call__(self, image, bboxes, class_labels):
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)