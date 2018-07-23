import torch
from torch.distributions import Normal
import torchvision.utils as vutils
from torch.autograd import Variable as V
from dcgan_model import DC_Generator, get_coordinates
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--latdim", type=int, default=32)
    parser.add_argument('--G', default='G.pth', help="path to G (to continue training)")
    parser.add_argument("--bsize", type=int, default=64)
    return parser.parse_args()

args = parse_args()

torch.manual_seed(1211)

def generate(generator, x_dim = 500, y_dim = 500):
    
    x, y, r = get_coordinates(28, 28, batch_size = 64)
    x_b, y_b, r_b = get_coordinates(x_dim, y_dim, batch_size = 1)
    
    
    seed_dist = Normal(V(torch.zeros(args.bsize, args.latdim)), 
                               V(torch.ones(args.bsize, args.latdim)))
    
    seed_sample = seed_dist.sample()
    
    seed = torch.bmm(torch.ones(args.bsize, 28 * 28, 1), seed_sample.unsqueeze(1))
    
    
    fake = generator(x, y, r, seed)
    
    vutils.save_image(fake.data,
            'images/GAN_generated_samples.png',
            normalize=True)

    i = input("Which sample to blow up?")
    
    seed = torch.bmm(torch.ones(args.bsize, x_dim * y_dim, 1), seed_sample.unsqueeze(1))
    
    print(x_b.size(), y_b.size(), r_b.size(), seed[int(i)].unsqueeze(0))
    
    fake = generator(x_b, y_b, r_b, seed[int(i)].unsqueeze(0))
    
    vutils.save_image(fake.data,
            'images/GAN_blowup.png',
            normalize=True)
    
if __name__ == '__main__':
    
    assert args.G != '', "Generator Model must be provided!"
    
    G = DC_Generator(x_dim = 28, y_dim = 28, z_dim=args.latdim, batch_size = args.bsize)
    G.load_state_dict(torch.load("G.pth", map_location=lambda storage, loc: storage))
    
    generate(generator = G)
    print("Generated sample saved to GAN_blowup.png")