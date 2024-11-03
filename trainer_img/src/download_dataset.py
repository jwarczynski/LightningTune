from torchvision import transforms, datasets

datasets.FashionMNIST("data", train=True, download=True)
datasets.FashionMNIST("data", train=False, download=True)