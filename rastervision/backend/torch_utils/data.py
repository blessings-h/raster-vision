from os.path import join
from collections import defaultdict
import random
import glob

from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import torchvision
import cv2

from rastervision.utils.files import (file_to_json)
from rastervision.backend.torch_utils.boxlist import BoxList

from albumentations import (
    Posterize,
    IAAAffine,
    RandomCropNearBBox,
    IAASharpen,
    HorizontalFlip,
    Equalize,
    Rotate, 
    RandomContrast,
    Solarize,
    IAASharpen,
    ToGray,
    Cutout,
    RandomBrightness,
    Resize,
    Compose,
    OneOf,
    BboxParams
)

from albumentations.pytorch import ToTensor

class DataBunch():
    def __init__(self, train_ds, train_dl, valid_ds, valid_dl, label_names):
        self.train_ds = train_ds
        self.train_dl = train_dl
        self.valid_ds = valid_ds
        self.valid_dl = valid_dl
        self.label_names = label_names

    def __repr__(self):
        rep = ''
        if self.train_ds:
            rep += 'train_ds: {} items\n'.format(len(self.train_ds))
        if self.valid_ds:
            rep += 'valid_ds: {} items\n'.format(len(self.valid_ds))
        if self.test_ds:
            rep += 'test_ds: {} items\n'.format(len(self.test_ds))
        rep += 'label_names: ' + ','.join(self.label_names)
        return rep

def collate_fn(data):
    x = [d[0].unsqueeze(0) for d in data]
    boxLists = [(torch.tensor(d[1]).float(), torch.tensor(d[2])) for d in data]
    y = []
    
    for item in boxLists:
        if len(item[0]) > 0:
            boxes = item[0]
            labels = item[1]
            x_min, y_min, w, h = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
            boxes = torch.cat([y_min, x_min, y_min+h, x_min+w], dim=1).float()
            y.append(BoxList(boxes, labels=labels))
        else:
            y.append(BoxList(torch.empty((0, 4)).float(), labels=torch.empty((0,))))

    return (torch.cat(x).float(), y)


class CocoDataset(Dataset):
    def __init__(self, img_dir, annotation_uris, transforms=None):
        self.img_dir = img_dir
        self.annotation_uris = annotation_uris
        self.transforms = transforms

        self.imgs = []
        self.img2id = {}
        self.id2img = {}
        self.id2boxes = defaultdict(lambda: [])
        self.id2labels = defaultdict(lambda: [])
        self.label2name = {}
        for annotation_uri in annotation_uris:
            ann_json = file_to_json(annotation_uri)
            for img in ann_json['images']:
                self.imgs.append(img['file_name'])
                self.img2id[img['file_name']] = img['id']
                self.id2img[img['id']] = img['file_name']
            for ann in ann_json['annotations']:
                img_id = ann['image_id']
                box = ann['bbox']
                label = ann['category_id']
#                box = [box[1], box[0], box[1] + box[3], box[0] + box[2]] ####
                self.id2boxes[img_id].append(box)
                self.id2labels[img_id].append(label)
                
        random.seed(1234)
        random.shuffle(self.imgs)
#        self.id2boxes = dict([(id, torch.cat(boxes).float())
        self.id2boxes = dict([(id, boxes)
                              for id, boxes in self.id2boxes.items()])
#        self.id2labels = dict([(id, torch.tensor(labels))
        self.id2labels = dict([(id, labels)
                               for id, labels in self.id2labels.items()])

    def __getitem__(self, ind):
        img_fn = self.imgs[ind]
        img_id = self.img2id[img_fn]
        img = np.array(Image.open(join(self.img_dir, img_fn)))
        
        boxes = dict()
        labels = dict()
        
        if img_id in self.id2boxes:
            boxes, labels = self.id2boxes[img_id], self.id2labels[img_id]
#        else:
#            boxes, labels = torch.empty((0, 4)), torch.empty((0, )).long()
#            boxlist = BoxList(
#                torch.empty((0, 4)), labels=torch.empty((0, )).long())
        
        if self.transforms:
            out = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = out['image']
#            boxes = torch.tensor(out['bboxes'])
#            labels = torch.tensor(out['labels'])
            boxes = out['bboxes']
            labels = out['labels']
        
#        if len(boxes) > 0:
#            x, y, w, h = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
#            boxes = torch.cat([y, x, y+h, x+w], dim=1)
#            boxlist = BoxList(boxes, labels=labels)
#        else:
#            boxlist = BoxList(torch.empty((0, 4)), labels=torch.empty((0,)))
#        return (img, boxlist)
        return (img, boxes, labels)

    def __len__(self):
        return len(self.imgs)
    
    

def get_label_names(coco_path):
    categories = file_to_json(coco_path)['categories']
    label2name = dict([(cat['id'], cat['name']) for cat in categories])
    labels = ['background'
              ] + [label2name[i] for i in range(1,
                                                len(label2name) + 1)]
    return labels


def build_databunch(data_dir, img_sz, batch_sz):
    # TODO This is to avoid freezing in the middle of the first epoch. Would be nice
    # to fix this.
    num_workers = 0

    train_dir = join(data_dir, 'train')
    train_anns = glob.glob(join(train_dir, '*.json'))
    valid_dir = join(data_dir, 'valid')
    valid_anns = glob.glob(join(valid_dir, '*.json'))

    label_names = get_label_names(train_anns[0])
    aug_transforms = [HorizontalFlip(),
                      ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.8),
                      Equalize(p=0.8),
                      RandomContrast()
                     ]

    
    aug_transforms = [ OneOf(
            [ Posterize(p = 0.8), IAAAffine(translate_px={"x": 10, "y": 0}, p=1.0),], 
            [ RandomCropNearBBox(p=0.2), IAASharpen(p=0.5),], 
            [ Rotate(p=0.6), Rotate(p=0.8),],  
            [ Equalize(p=0.8), RandomContrast(p=0.2),], 
            [ Solarize(p=0.2), IAAAffine(translate_px={"x": 0, "y": 10}, p=0.2),], 
            [ IAASharpen(p=0.0), ToGray(p=0.4),], 
            [ Equalize(p=1.0), IAAAffine(translate_px={"x": 0, "y": 10}, p=0.2),], 
            [ Posterize(p=0.8), Rotate(p=0.0),], 
            [ RandomContrast(p=0.6), Rotate(p=1.0),], 
            [ Equalize(p=0.0), Cutout(p=0.8),], 
            [ RandomBrightness(p=1.0), IAAAffine(translate_px={"x": 0, "y": 10}, p=0.2),], 
            [ RandomContrast(p=0.0), IAAAffine(shear=60.0, p=0.8),], 
            [ RandomContrast(p=0.8), RandomContrast(p=0.2),], 
            [ Roatate(p=1.0), Cutout(p=1.0),], 
            [ Solarize(p=0.8), Equalize(p=0.8),],          
            p = 1.0)
    ]            
    
    
    transforms = [Resize(img_sz, img_sz), ToTensor()]
    aug_transforms.extend(transforms)
    
    bbox_params = BboxParams(format='coco', min_area=0., min_visibility=0.2, label_fields=['labels'])
    aug_transforms = Compose(aug_transforms, bbox_params=bbox_params)
    transforms = Compose(transforms, bbox_params=bbox_params)
    
    train_ds = CocoDataset(train_dir, train_anns, transforms=aug_transforms)
    valid_ds = CocoDataset(valid_dir, valid_anns, transforms=transforms)
    train_ds.label_names = label_names
    valid_ds.label_names = label_names

    train_dl = DataLoader(
        train_ds,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)
    valid_dl = DataLoader(
        valid_ds,
        collate_fn=collate_fn,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)
    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, label_names)
