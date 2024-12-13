# data.py

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


# Initializes the data handler with configurations for dataset preprocessing and loading
# Goal: Converts MNIST images to tensors and normalizes them
class DataHandler:
    def __init__(self, data_root, mnist_mean, mnist_std_dev, num_train_samples, num_test_samples, batch_size):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std_dev),
        ])
        self.data_root = data_root
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.batch_size = batch_size

    # Select a subset of the dataset based on the number of samples
    def load_data(self, train=True):
        dataset = MNIST(root=self.data_root, train=train, download=True, transform=self.transform)
        if train:
            subset = Subset(dataset, list(range(self.num_train_samples)))
        else:
            subset = Subset(dataset, list(range(self.num_test_samples)))
        return subset

    # Creates and returns DataLoaders for training and testing datasets.
    def get_dataloaders(self):
        train_subset = self.load_data(train=True)
        test_subset = self.load_data(train=False)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

        return train_loader, test_loader
