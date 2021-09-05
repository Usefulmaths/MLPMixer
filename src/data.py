import pytorch_lightning as pl
import torch
import torchvision


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size

    def setup(self, stage=None):
        dataset = torchvision.datasets.MNIST(
            self._data_dir,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        self._train_dataset, self._val_dataset = torch.utils.data.random_split(
            dataset, [50000, 10000]
        )

        self._test_dataset = torchvision.datasets.MNIST(
            self._data_dir,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_dataset, batch_size=self._batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_dataset, batch_size=self._batch_size, shuffle=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test_dataset, batch_size=self._batch_size, shuffle=True
        )
