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
        - subset_name (str): string preceding the .nii extension. 
            Only images with this substring will be used. Defaults to ""
        - batch_size (int): size of the batch. Defaults to 16
        - num_workers (int): number of workers for the dataloaders. Defaults to 0
        - train_val_ratio (float): ratio of the data to use in validation. Defaults to 0.7
        - test_has_labels (bool); indicates whether or not to look for and download labels for test dataset
    """
    def __init__(self,
                 task: str,
                 subset_name: str = "",
                 batch_size: int = 16,
                 num_workers: int = 0,
                 train_val_ratio: float = 0.7,
                 test_has_labels: bool = True):
        super().__init__()
        self.task = task
        self.subset_name = subset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_dir = Path(task)
        self.train_val_ratio = train_val_ratio
        self.test_has_labels = test_has_labels
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
            return sorted(p for p in d.glob(f'*{self.subset_name}.nii*')
                          if not p.name.startswith('.'))

        image_training_paths = get_niis(self.dataset_dir / 'imagesTr')
        label_training_paths = get_niis(self.dataset_dir / 'labelsTr')
        image_test_paths = get_niis(self.dataset_dir / 'imagesTs')
        label_test_paths = []

        if self.test_has_labels:
            label_test_paths = get_niis(self.dataset_dir / 'labelsTs')

        return image_training_paths, label_training_paths, image_test_paths, label_test_paths

    def prepare_data(self):
        image_training_paths, label_training_paths, image_test_paths, label_test_paths = self.download_data(
        )
        self.subjects = []
        self.test_subjects = []

        for image_path, label_path in zip(image_training_paths,
                                          label_training_paths):
            subject = tio.Subject(image=tio.ScalarImage(image_path),
                                  label=tio.LabelMap(label_path))

            self.subjects.append(subject)

        if self.test_has_labels:
            for image_path in zip(image_test_paths, label_test_paths):
                subject = tio.Subject(image=tio.ScalarImage(image_path),
                                      label=tio.LabelMap(label_path))

                self.test_subjects.append(subject)
        else:
            for image_path in image_test_paths:
                subject = tio.Subject(image=tio.ScalarImage(image_path))

                self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.ZNormalization(),
            # tio.CropOrPad(
            #     self.get_max_shape(self.subjects + self.test_subjects)),
            # tio.EnsureShapeMultiple(8),  # for the U-Net
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
        self.transform = tio.Compose([self.preprocess])  #, augment])

        self.train_set = tio.SubjectsDataset(train_subjects,
                                             transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects,
                                           transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects,
                                            transform=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          self.batch_size,
                          num_workers=self.num_workers)
