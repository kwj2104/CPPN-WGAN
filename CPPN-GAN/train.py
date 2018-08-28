import torch
from torch.autograd import Variable as V
import argparse
from util import load_mnist
from torch.distributions import Normal
from dcgan_model import DC_Discriminator, DC_Generator, get_coordinates
import torchvision.utils as vutils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--latdim", type=int, default=32)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--beta1", type=float, default=.5)
    parser.add_argument("--lr_g", type=float, default=0.01)
    parser.add_argument("--lr_d", type=float, default=0.01)
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--handicap", type=int, default=3)
    parser.add_argument("--th_high", type=int, default=100)
    parser.add_argument("--th_low", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=1111)
    return parser.parse_args()

args = parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(args.random_seed)
if use_cuda:
    torch.cuda.manual_seed(args.random_seed)

def train_gan(G, D, train_loader, optim_disc, optim_gen, seed_dist, x, y, r, epoch):
    iter_counter = 0
    for t in train_loader:
        img, _ = t
        if img.size()[0] < args.bsize : continue
        
        img = img.to(device)

        optim_disc.zero_grad()
        optim_gen.zero_grad()
        
        # Grad generator
        # Generator maximizes log probability of discriminator being mistaken
        # This trick deals with generator's vanishing gradients
        # See NIPS 2016 Tutorial: Generative Adversarial Networks 
        # Give handicap to generator by giving it extra steps to adjust weights
        for i in range(args.handicap):
            seed = torch.bmm(torch.ones(args.bsize, 28 * 28, 1), seed_dist.sample().unsqueeze(1))
            seed = seed.to(device)
            
            x_fake = G(x, y, r, seed)
            d = D(x_fake)
            loss_c = -d.log().mean()
            loss_c.backward()
            
            optim_gen.step() 
        
        # Grad Discriminator
        # Grad real
        # -E[log(D(x))]
        d = D(img)
        loss_a = -d.log().mean()

        # Grad fake
        # -E[log(1 - D(G(z)) )]
        seed = torch.bmm(torch.ones(args.bsize, 28 * 28, 1), seed_dist.sample().unsqueeze(1))
        seed = seed.to(device)
        
        x_fake = G(x, y, r, seed)
        d = D(x_fake.detach())     
        loss_b = -(1 - d + 1e-10).log().mean()
        
        disc_loss = loss_a + loss_b
        
        if loss_c < args.th_high and disc_loss > args.th_low:
            disc_loss.backward()
            optim_disc.step()
        
        # Output loss
        if iter_counter % 10 == 0:
            print("Iteration: {} Discriminator Loss: {} Generator Loss: {}".format(iter_counter, disc_loss, loss_c))
        
        # Save samples
        if iter_counter % 800 == 0:
            seed = torch.bmm(torch.ones(args.bsize, 28 * 28, 1), seed_dist.sample().unsqueeze(1))
            seed = seed.to(device)
            x_fake = G(x, y, r, seed)
            vutils.save_image(x_fake.data,
            'GAN_samples_' + str(epoch) + '.png',
            normalize=True)
        
        iter_counter += 1
       
 

if __name__ == "__main__":
    train_loader = load_mnist(args.bsize)
    
    G = DC_Generator(x_dim = 28, y_dim = 28, z_dim=args.latdim, batch_size = args.bsize).to(device)
    D = DC_Discriminator(1, args.ndf).to(device)
    optim_gen = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))    
    optim_disc = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))
    
    
    seed_distribution = Normal(V(torch.zeros(args.bsize, args.latdim)), 
                               V(torch.ones(args.bsize, args.latdim)))
    
        
    #init cppn coordinates
    x, y, r = get_coordinates(28, 28, batch_size=args.bsize)
    x = x.to(device)
    y = y.to(device)
    r = r.to(device)
    
    #Train
    print("Training..")
    for epoch in range(args.epoch):
        print("Training Epoch {}".format(epoch + 1))
        train_gan(G, D, train_loader, optim_disc, optim_gen, seed_distribution, x, y, r, epoch=epoch)
        
    torch.save(G.state_dict(), "G.pth")
    torch.save(D.state_dict(), "D.pth")