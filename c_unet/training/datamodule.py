import torchio as tio
import pytorch_lightning as pl
import numpy as np

from torch.utils.data import random_split, DataLoader
from pathlib import Path


class DataModule(pl.LightningDataModule):
    """
    Data structure to use when traning with Pytorch Lightning

    Args:
        - task (str): name of the task to perform. Corresponds to the name of the folder where the data is stored.
        - batch_size (int): size of the batch. Defaults to 16
        - train_val_ratio (float): ratio of the data to use in validation. Defaults to 0.7
    """
    def __init__(self, 
                    task: str,
                    batch_size: int = 16,
                    train_val_ratio: float = 0.7):
        super().__init__()
        self.task = task
        self.batch_size = batch_size
        self.dataset_dir = Path(task)
        self.train_val_ratio = train_val_ratio
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
    
    def get_max_shape(self, subjects):
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.get_first_image().spatial_shape for s in dataset])
        return shapes.max(axis=0)
    
    def download_data(self):

        def get_niis(d):
            return sorted(p for p in d.glob('*standard.nii*') if not p.name.startswith('.'))

        image_training_paths = get_niis(self.dataset_dir / 'imagesTr')
        label_training_paths = get_niis(self.dataset_dir / 'labelsTr')
        image_test_paths = get_niis(self.dataset_dir / 'imagesTs')

        return image_training_paths, label_training_paths, image_test_paths

    def make_image_group_compatible(self, subject):
        image_group = subject['image_group'][tio.DATA]
        image_group = image_group.unsqueeze(0)
        subject['image_group'][tio.DATA] = image_group
        return subject

    def prepare_data(self):
        image_training_paths, label_training_paths, image_test_paths = self.download_data()
        self.subjects = []
        self.test_subjects = []

        for image_path, label_path in zip(image_training_paths, label_training_paths):
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                image_group=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path)
            )

            # Adding dimension to make inputs usable with group convolutions
            subject = self.make_image_group_compatible(subject)

            self.subjects.append(subject)
        
        for image_path in image_test_paths:
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                image_group=tio.ScalarImage(image_path)
            )
            subject = self.make_image_group_compatible(subject)

            self.test_subjects.append(subject)
    
    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad(self.get_max_shape(self.subjects + self.test_subjects)),
            tio.EnsureShapeMultiple(8),  # for the U-Net
            tio.OneHot(),
        ])
        return preprocess
    
    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomAffine(),
            tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25),
        ])
        return augment

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=10)