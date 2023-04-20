from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset

import fiftyone
from PIL import Image

NUM_WORKERS = 4


class TensorImgSet(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        self.imgs = tensors[0]
        self.targets = tensors[1]
        self.tensors = tensors
        self.transform = transform
        self.len = len(self.imgs)

    def __getitem__(self, index):
        x = self.imgs[index]
        if self.transform:
            x = self.transform(x)
        y = self.targets[index]
        return x, y

    def __len__(self):
        return self.len

class FiftyOneTorchDataset(Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.
    
    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for 
            training or testing
        transforms (None): a list of PyTorch transforms to apply to images 
            and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset 
            that contains the desired labels to load
        classes (None): a list of class strings that are used to define the 
            mapping between class names and indices. If None, it will use 
            all classes present in the given fiftyone_dataset.
    """

    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="positive_labels",
        classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.default_classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        img = Image.open(img_path).convert("RGB")

        try:
            labels = sample[self.gt_field].classifications
            print(f'{len(labels)}')
        except:
            labels = sample['negative_labels'].classifications
            print(f'only negative {len(labels)}')
        
        label = labels[0].label
        target = self.labels_map_rev[label]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

def get_open_images(num_classes=100, dataset_dir="./data", batch_size=128):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    fo_trainset = fiftyone.zoo.load_zoo_dataset("open-images-v7", split="train", label_types=["classifications"],max_samples=100)
    trainset = FiftyOneTorchDataset(fo_trainset,train_transform)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               num_workers=NUM_WORKERS,
                                               pin_memory=True, shuffle=True)

    fo_testset = fiftyone.zoo.load_zoo_dataset("open-images-v7", split="validation", label_types=["classifications"],max_samples=100)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = FiftyOneTorchDataset(fo_testset,test_transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              num_workers=NUM_WORKERS,
                                              pin_memory=True, shuffle=False)
    
    return train_loader, test_loader

def load_cifar_10_1():
    # @article{recht2018cifar10.1,
    #  author = {Benjamin Recht and Rebecca Roelofs and Ludwig Schmidt
    #  and Vaishaal Shankar},
    #  title = {Do CIFAR-10 Classifiers Generalize to CIFAR-10?},
    #  year = {2018},
    #  note = {\url{https://arxiv.org/abs/1806.00451}},
    # }
    # Original Repo: https://github.com/modestyachts/CIFAR-10.1
    data_path = Path(__file__).parent.joinpath("cifar10_1")
    label_filename = data_path.joinpath("v6_labels.npy").resolve()
    imagedata_filename = data_path.joinpath("v6_data.npy").resolve()
    print(f"Loading labels from file {label_filename}")
    labels = np.load(label_filename)
    print(f"Loading image data from file {imagedata_filename}")
    imagedata = np.load(imagedata_filename)
    return imagedata, torch.Tensor(labels).long()


def get_cifar(num_classes=100, dataset_dir="./data", batch_size=128,
              use_cifar_10_1=False):

    if num_classes == 10:
        print("Loading CIFAR10...")
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        print("Loading CIFAR100...")
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = dataset(root=dataset_dir, train=True,
                       download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Use the normal cifar 10 testset or a new one to test true generalization
    if use_cifar_10_1 and num_classes == 10:
        imagedata, labels = load_cifar_10_1()
        testset = TensorImgSet((imagedata, labels), transform=test_transform)
    else:
        testset = dataset(root=dataset_dir, train=False,
                          download=True,
                          transform=test_transform)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               num_workers=NUM_WORKERS,
                                               pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              num_workers=NUM_WORKERS,
                                              pin_memory=True, shuffle=False)
    return train_loader, test_loader
