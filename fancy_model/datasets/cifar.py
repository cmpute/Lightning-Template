from torchvision import transforms
from torchvision.datasets import CIFAR10

class CIFAR10Dataset:
    """ Dummy dataset based on CIFAR10 """
    def __init__(self, path, split="train") -> None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self._dataset = CIFAR10(
            root=path,
            train=split == "train",
            transform=transform,
            download=True
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        img, _ = self._dataset[index]
        return img # we don't need the labels
