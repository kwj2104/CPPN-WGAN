import os, sys
sys.path.append(os.getcwd())
import tflib as lib
from tflib import cifar10
import torch
import torch.autograd as autograd
import gan_cppn_mnist as mnist
import gan_cppn_cifar10 as cf10
import gan_cppn_casia as casia
import numpy as np
import imageio
import random

torch.manual_seed(111)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
    
BATCH_SIZE = 64 # Batch size
LATENT_DIM = 128

'''
Create image samples and gif interpolations of trained models
'''


def interpolate(state_dict, generator, preview = True, interpolate = False, large_sample=False,
                 img_size=28, img_channel=1, large_dim=1024,
                 samples=[random.randint(0, BATCH_SIZE - 1), random.randint(0, BATCH_SIZE - 1)]):
    
    """
    Args:
        
    state_dict: saved copy of trained params
    generator: generator model
    preview: show preview of images in grid form in original size (to pick which to blow up)
    interpolate: create interpolation gif
    large_sample: create a large sample of an individual picture
    img_size: size of your input samples, e.g. 28 for MNIST
    img_channel: number of color channels, 3 for cifar
    large_dim: dimension to blow up samples to for interpolation
    samples: indices of the samples you want to interpolate
    """
    
    x_d = img_size
    y_d = img_size
    c_d = img_channel
    position = 8
    x, y, r = mnist.get_coordinates(x_d, y_d, batch_size=BATCH_SIZE)    
    x_large = large_dim
    y_large = large_dim
    
    
    generator_int = generator
    generator_int.load_state_dict(torch.load(state_dict, map_location=lambda storage, loc: storage))
    
    noise = torch.randn(BATCH_SIZE, LATENT_DIM)
    
    if preview:
        
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise, volatile=True)
        
        ones = torch.ones(BATCH_SIZE, x_d * y_d, c_d)
        if use_cuda:
            ones = ones.cuda()
                
        seed = torch.bmm(ones, noisev.unsqueeze(1))
        
        gen_imgs = generator_int(x, y, r, seed)
        
        gen_imgs = gen_imgs.cpu().data.numpy()
    
        lib.save_images.save_images(
            gen_imgs,
            'generated_img/samples.png'
            
        )

    elif large_sample:
        noisev = autograd.Variable(noise[position], volatile=True)
        ones = torch.ones(1, x_large * y_large, 1)
        seed = torch.bmm(ones, noisev.unsqueeze(0).unsqueeze(0))
        x, y, r = mnist.get_coordinates(x_large, y_large, batch_size=1)
        
        gen_imgs = generator_int(x, y, r, seed)
        gen_imgs = gen_imgs.cpu().data.numpy()
        
        lib.save_images.save_images(
            gen_imgs,
            'generated_img/large_sample.png'
        )
    elif interpolate:
               
        nbSteps = 10
        alphaValues = np.linspace(0, 1, nbSteps)
        images = []
        
        noisev = autograd.Variable(noise[position], volatile=True)
        ones = torch.ones(1, x_large * y_large, 1)
        seed = torch.bmm(ones, noisev.unsqueeze(0).unsqueeze(0))
        x, y, r = mnist.get_coordinates(x_large, y_large, batch_size=1)
        
        samples.append(samples[0])
        
        for i in range(len(samples) - 1):  
            for alpha in alphaValues:                    
                vector = noise[samples[i]].unsqueeze(0)*(1-alpha) + noise[samples[i + 1]].unsqueeze(0)*alpha
                gen_imgs = generator_int(x, y, r, vector)
                
                if c_d == 3:
                    gen_img_np = np.transpose(gen_imgs.data[0].numpy())
                elif c_d == 1:
                    gen_img_np = np.transpose(gen_imgs.data.numpy()).reshape(x_large, y_large, -1)
                    
                images.append(gen_img_np)
            
        
        imageio.mimsave('generated_img/movie.gif', images)
        
        
        
if __name__ == "__main__":
    
    #Casia 
    G = casia.Generator(x_dim = 64, y_dim = 64, z_dim=LATENT_DIM, batch_size = BATCH_SIZE)
    interpolate("G-cppn-wgan_casia.pth", G, img_size=64, preview=False, large_sample=False, img_channel = 1, 
                interpolate=True, large_dim=32, samples=[4, 7, 9, 3])
    
#    #MNIST
#    G = mnist.Generator(x_dim = 28, y_dim = 28, z_dim=LATENT_DIM, batch_size = BATCH_SIZE)
#    interpolate("G-cppn-wgan_mnist.pth", G, img_size=28, preview=False, large_sample=False, img_channel = 1, 
#                interpolate=True, large_dim=50, samples=[1, 2, 3])
    #CIFAR10
#    G = cifar10.Generator(x_dim = 32, y_dim = 32, z_dim=LATENT_DIM, batch_size = BATCH_SIZE)
#    interpolate("G-cppn-wgan_cifar10.pth", G, img_size=32, preview=False, large_sample=False, img_channel = 3, 
#                interpolate=True, large_dim=500, samples=[1, 2, 3])
