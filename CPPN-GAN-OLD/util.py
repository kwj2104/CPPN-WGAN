import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_mnist(bsize):
    
    dataset = datasets.MNIST(root = '../data/',
                         transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]),
                          download = True)


    loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = bsize,
                                     shuffle = True)

    return loader