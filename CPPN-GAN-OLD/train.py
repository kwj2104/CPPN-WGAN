import torch
from torch.autograd import Variable as V
import argparse
from util import load_mnist
from torch.distributions import Normal
from dcgan_model import DC_Discriminator, DC_Generator, get_coordinates
import torchvision.utils as vutils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)
    parser.add_argument("--epoch", type=int, default=10)
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
    return parser.parse_args()

args = parse_args()

#torch.manual_seed(1111)
#if args.devid >= 0:
#    torch.cuda.manual_seed(1111)

def train_gan(G, D, train_loader, optim_disc, optim_gen, seed_dist, x, y, r):
    iter_counter = 0
    for t in train_loader:
        img, _ = t
        if img.size()[0] < args.bsize : continue
        if args.devid >= 0:
            img = img.cuda()

        optim_disc.zero_grad()
        optim_gen.zero_grad()
        
        # Grad generator
        # Generator maximizes log probability of discriminator being mistaken
        # This trick deals with generator's vanishing gradients
        # See NIPS 2016 Tutorial: Generative Adversarial Networks 
        # Give handicap to generator by giving it extra steps to adjust weights
        for i in range(args.handicap):
            seed = torch.bmm(torch.ones(args.bsize, 28 * 28, 1), seed_dist.sample().unsqueeze(1))
            if args.devid >= 0:
                seed = seed.cuda()
            
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
        if args.devid >= 0:
            seed = seed.cuda()
        
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
            if args.devid >= 0:
                seed = seed.cuda()
            x_fake = G(x, y, r, seed)
            vutils.save_image(x_fake.data,
            'images/GAN_samples.png',
            normalize=True)
        
        iter_counter += 1
       
 

if __name__ == "__main__":
    train_loader = load_mnist(args.bsize)
    x_d = 28
    y_d = 28
    
    G = DC_Generator(x_dim = x_d, y_dim = y_d, z_dim=args.latdim, batch_size = args.bsize)
    D = DC_Discriminator(1, args.ndf)
    optim_gen = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))    
    optim_disc = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))
    
    
    seed_distribution = Normal(V(torch.zeros(args.bsize, args.latdim)), 
                               V(torch.ones(args.bsize, args.latdim)))
    
    if args.devid >= 0:
        G.cuda()
        D.cuda()
        
    #init cppn coordinates
    x, y, r = get_coordinates(x_d, y_d, batch_size=args.bsize)
    if args.devid >= 0:
        x = x.cuda()
        y = y.cuda()
        r = r.cuda()
    
    #Train
    print("Training..")
    for epoch in range(args.epoch):
        print("Training Epoch {}".format(epoch + 1))
        train_gan(G, D, train_loader, optim_disc, optim_gen, seed_distribution, x, y, r)
        if epoch > 1:
            x_b, y_b, r_b = get_coordinates(1080, 1080, 1)
            if args.devid >= 0:
                x_b = x_b.cuda()
                y_b = y_b.cuda()
                r_b = r_b.cuda()
            seed = torch.bmm(torch.ones(args.bsize, 1080 * 1080, 1), seed_distribution.sample().unsqueeze(1))
            if args.devid >= 0:
                seed = seed.cuda()
            fake = G(x_b, y_b, r_b, seed[0])
            vutils.save_image(fake.data,
            'images/GAN_blowup.png',
            normalize=True)
        
    torch.save(G.state_dict(), "G.pth")
    torch.save(D.state_dict(), "D.pth")