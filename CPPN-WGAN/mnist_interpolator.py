import os, sys
sys.path.append(os.getcwd())
import tflib.save_images as save_images
import torch
import torch.autograd as autograd
import numpy as np
import imageio
import random
import argparse
import gan_cppn_mnist as mnist

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latdim", type=int, default=128)
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--random_seed", type=int, default=1111)
    parser.add_argument("--sample", type=bool, default=False)
    parser.add_argument("--large_dim", type=int, default=500)
    parser.add_argument("--large_sample", type=int, default=-1)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument('--interpolate', nargs='*', type=int)
    
    return parser.parse_args()

args = parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(args.random_seed)
if use_cuda:
    torch.cuda.manual_seed(args.random_seed)

'''
Create image samples and gif interpolations of trained models
'''


def interpolate(state_dict, generator, preview = True, interpolate = False, large_sample=False,
                 img_size=28, img_channel=1, large_dim=1024,
                 samples=[random.randint(0, args.bsize - 1), random.randint(0, args.bsize - 1)]):
    
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
    #position = 2
    x, y, r = mnist.get_coordinates(x_d, y_d, batch_size=args.bsize)    
    x_large = large_dim
    y_large = large_dim
    
    
    generator_int = generator
    generator_int.load_state_dict(torch.load(state_dict, map_location=lambda storage, loc: storage))
    
    noise = torch.randn(args.bsize, args.latdim)
    
    if preview:
        
        noise = noise.to(device)
        noisev = autograd.Variable(noise, volatile=True)
        
        ones = torch.ones(args.bsize, x_d * y_d, c_d)
        ones = ones.to(device)
                
        seed = torch.bmm(ones, noisev.unsqueeze(1))
        
        gen_imgs = generator_int(x, y, r, seed)
        
        gen_imgs = gen_imgs.cpu().data.numpy()
    
        save_images.save_images(
            gen_imgs,
            'generated_img/samples.png'
            
        )

    if large_sample >= 0:
        
        assert args.sample < args.bsize, "Sample position is out of bounds"
        
        noise = noise.to(device)
        noisev = autograd.Variable(noise[large_sample], volatile=True)
        ones = torch.ones(1, x_large * y_large, 1).to(device)
        seed = torch.bmm(ones, noisev.unsqueeze(0).unsqueeze(0))
        x, y, r = mnist.get_coordinates(x_large, y_large, batch_size=1)
        
        gen_imgs = generator_int(x, y, r, seed)
        gen_imgs = gen_imgs.cpu().data.numpy()
        
        save_images.save_images(
            gen_imgs,
            'generated_img/large_sample.png'
        )
    if interpolate:
               
        nbSteps = args.frames
        alphaValues = np.linspace(0, 1, nbSteps)
        images = []
        
        noise = noise.to(device)
        noisev = autograd.Variable(noise[samples[0]], volatile=True)
        ones = torch.ones(1, x_large * y_large, 1).to(device)
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
    

    if not os.path.exists("generated_img"):
        os.makedirs("generated_img")
    
    interp = False
    interp_list = []
    for arg in args._get_kwargs():
        if arg[0] == 'interpolate' and arg[1] != None:
            interp = True
            interp_list = arg[1]
    G = mnist.Generator(x_dim = 28, y_dim = 28, z_dim=args.latdim, batch_size = args.bsize)
    #G = model.DC_Generator(x_dim = 28, y_dim = 28, z_dim=args.latdim, batch_size = args.bsize)
    interpolate("G.pth", G, img_size=28, preview=args.sample, large_sample=args.large_sample, img_channel = 1, 
                interpolate=interp, large_dim=args.large_dim, samples=interp_list)