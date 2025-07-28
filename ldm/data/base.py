import bisect
import random
from matplotlib import axis
import torch
from torch.utils.data import Dataset
from abc import abstractmethod
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
import albumentations
from PIL import Image
import numpy as np
import cv2
from pytorch_lightning import seed_everything
import json
import importlib

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class ConcatDataset(Dataset):
    def __init__(self, data_cfgs):
        self.datasets = torch.utils.data.ConcatDataset([instantiate_from_config(cfg) for cfg in data_cfgs])

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        return self.datasets.__getitem__(item)


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx

class ImagePathsSTv1(Dataset):
    def __init__(self, paths, size=384, crop_size=256, random_augment=False, mode=None,smallest_max_size=True, labels=None):
        assert mode in ["content","style"]
        self.mode = mode
        self.size = size
        self.crop_size = crop_size or size
        self.random_augment = random_augment
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size) if smallest_max_size else \
                albumentations.Resize(height=self.size, width=self.size)
            self.cropper = \
                albumentations.RandomCrop(height=self.crop_size, width=self.crop_size) if random_augment else \
                    albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            transforms = [self.rescaler, self.cropper]
            if random_augment:
                self.flipper = albumentations.HorizontalFlip(p=0.5)
                transforms.append(self.flipper)

            self.preprocessor = albumentations.Compose(transforms=transforms)
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        if self.mode =="content":
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_image=(gray_image/127.5 - 1.0).astype(np.float32)
            image = (image/127.5 - 1.0).astype(np.float32)
            return image,gray_image
        else:
            image = (image/127.5 - 1.0).astype(np.float32)
            return image

    def __getitem__(self, i):
        example = dict()
        if self.mode =="content":
            example["content"],example['gray'] = self.preprocess_image(self.labels["file_path_"][i])
            return example
        else:
            example["style"] = self.preprocess_image(self.labels["file_path_"][i])
            return example
        


class ImagePaths(Dataset):
    def __init__(self, paths, size=384, crop_size=256, random_augment=False, smallest_max_size=True, labels=None):
        self.size = size
        self.crop_size = crop_size or size
        self.random_augment = random_augment

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size) if smallest_max_size else \
                albumentations.Resize(height=self.size, width=self.size)
            self.cropper = \
                albumentations.RandomCrop(height=self.crop_size, width=self.crop_size) if random_augment else \
                    albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            transforms = [self.rescaler, self.cropper]
            if random_augment:
                self.flipper = albumentations.HorizontalFlip(p=0.5)
                transforms.append(self.flipper)

            self.preprocessor = albumentations.Compose(transforms=transforms)
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image=(gray_image/127.5 - 1.0).astype(np.float32)
        image = (image/127.5 - 1.0).astype(np.float32)
        return image,gray_image

    def __getitem__(self, i):
        example = dict()
        example["image"],example['gray'] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass
     