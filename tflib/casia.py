import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from pycasia.CASIA import CASIA 
from PIL import Image
import PIL


dataset = 'competition-gnt'

def make_square(im, min_size=64, fill_color='white', resize=32):
    #img = Image.fromarray(np.uint8(im * 255) , 'L')
    img = im
    x, y = img.size
    size = max(min_size, x, y)
    new_img = Image.new('L', (size, size), fill_color)
    new_img.paste(img, ((size - x) // 2, (size - y) // 2))
    new_img = new_img.resize([resize,resize],PIL.Image.ANTIALIAS)
    
    #return np.array(new_img.getdata())
    return new_img

class CasiaDataset(Dataset):
    def __init__(self, transform=None, data_dir='./data/', resize=32):
        
        #Download CASIA Dataset
        casia = CASIA(path=data_dir)
        #casia.get_dataset(dataset)
        
        self.images = []
        self.labels = []

        for image, label in casia.load_dataset(dataset):
                self.images.append(make_square(image, resize=resize))
                self.labels.append(label)

        
        self.transform=transform
    
    def __getitem__(self, index):
        img   = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
    def __len__(self):
        return len(self.images)

def load_casia(bsize, path, resize):
    dataset = CasiaDataset(transform=transforms.Compose([
                               transforms.ToTensor()
                           ]), data_dir=path, resize=resize)
    loader  = DataLoader(dataset, batch_size=bsize, shuffle=True)
    
    return loader

#casia_dataset = load_casia(bsize=1)

#i=0
#for img in casia_dataset:
#    if i==0:
#        imgplot = plt.imshow(make_square(np.squeeze(img[0][0].numpy())))
#        i = i + 1
    
#x = np.load('test_char.npy')
#im = make_square(x)
#print(im)



        
