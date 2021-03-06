import torchio as tio
import pytorch_lightning as pl
import numpy as np

from torch import Generator as Generator
from torch.utils.data import random_split, DataLoader
from pathlib import Path

from c_unet.training.HomogeniseLaterality import HomogeniseLaterality


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
                 test_has_labels: bool = False,
                 seed: int = 1):
        super().__init__()
        self.task = task
        self.subset_name = subset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_dir = Path(task)
        self.train_val_ratio = train_val_ratio
        self.test_has_labels = test_has_labels
        self.seed = seed
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def get_max_shape(self, subjects):
        """
        Get the maximum shape in every direction over the given list of subjects

        Args:
            subjects: list of tio.Subject
        Returns:
            Tuple with the maximum shape in each dimension
        """
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.get_first_image().spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def download_data(self):
        """
        Get the paths to all images

        Returns:
            Four lists with the training (and validation) images and labels paths, and test
            images and labels paths respectively
        """
        def get_niis(d):
            file_name = f'*.nii*'
            if self.subset_name:
                file_name = f'*{self.subset_name}*.nii*'
            return sorted(p for p in d.glob(file_name)
                          if not p.name.startswith('.'))

        image_training_paths = get_niis(self.dataset_dir / 'imagesTr')
        label_training_paths = get_niis(self.dataset_dir / 'labelsTr')
        image_test_paths = get_niis(self.dataset_dir / 'imagesTs')
        label_test_paths = []

        if self.test_has_labels:
            label_test_paths = get_niis(self.dataset_dir / 'labelsTs')

        return image_training_paths, label_training_paths, image_test_paths, label_test_paths

    def prepare_data(self):
        """
        Creates Subject instances with the image, label and laterality for training
        and validation subjects, and for test subjects
        """
        image_training_paths, label_training_paths, image_test_paths, label_test_paths = self.download_data(
        )
        self.subjects = []
        self.test_subjects = []

        for image_path, label_path in zip(image_training_paths,
                                          label_training_paths):
            subject = tio.Subject(
                image=tio.ScalarImage(image_path,
                                      filename=f"{image_path}".split('/')[-1]),
                label=tio.LabelMap(label_path,
                                   filename=f"{label_path}".split('/')[-1]),
                laterality="left" if "left" in str(image_path) else "right")

            self.subjects.append(subject)

        if self.test_has_labels:
            for image_path, label_path in zip(image_test_paths,
                                              label_test_paths):
                subject = tio.Subject(
                    image=tio.ScalarImage(
                        image_path, filename=f"{image_path}".split('/')[-1]),
                    label=tio.LabelMap(
                        label_path, filename=f"{label_path}".split('/')[-1]),
                    laterality="left"
                    if "left" in str(image_path) else "right")

                self.test_subjects.append(subject)
        else:
            for image_path in image_test_paths:
                subject = tio.Subject(image=tio.ScalarImage(
                    image_path, filename=f"{image_path}".split('/')[-1]),
                                      laterality="left" if "left"
                                      in str(image_path) else "right")

                self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        """
        Gets the composition of preprocessing transforms, which are applied on all subjects.
        They ensure a unique 'multiple of eight' shape for all of them, normalize the intensity
        values, and the laterality (left hippocampi are reversed to face right-ward), as well
        as the orientation. Labels are one-hot encoded.

        Returns:
            Composition of transformations
        """
        preprocess = tio.Compose([
            HomogeniseLaterality(
                from_laterality='left',
                axes='L',
            ),
            tio.ToCanonical(),
            tio.ZNormalization(),
            tio.CropOrPad(
                self.get_max_shape(self.subjects + self.test_subjects),
                mask_name="label",
            ),
            tio.EnsureShapeMultiple(
                8,
                method='pad',
            ),
            tio.OneHot(),
        ])
        return preprocess

    def get_augmentation_transform(self):
        """
        Gets the composition of augmentation transforms, which are applied on some training subjects
        randomly.

        Returns:
            Composition of transformations
        """
        augment = tio.Compose([
            tio.RandomAffine(p=0.1,
                             scales=0,
                             degrees=0,
                             translation=(0.05, 0.01, 0.05)),
            tio.RandomGamma(p=0.1, log_gamma=0.01),
            tio.RandomNoise(p=0.1, mean=0, std=(0, 0.01)),
        ])
        return augment

    def setup(self, stage=None):
        """
        Setups the data in three SubjectsDatasets (training, validation and test).
        The training data is split randomly between training and validation.
        """
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(
            self.subjects,
            splits,
            generator=Generator().manual_seed(self.seed))

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

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
