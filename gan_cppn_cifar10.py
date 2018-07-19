import os, sys
sys.path.append(os.getcwd())

import time
import tflib as lib
import tflib.cifar10
import tflib.plot
import math
import numpy as np

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = 'cifar-10-batches-py/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)
LATENT_DIM = 128 #Latent dimension of sample z

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

# ==================CPPN Modifications======================

def get_coordinates(x_dim = 28, y_dim = 28, scale = 8, batch_size = 1):
    '''
    calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
    '''
    n_points = x_dim * y_dim

    # creates a list of x_dim values ranging from -1 to 1, then scales them by scale
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5        
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    
    return torch.from_numpy(x_mat).float(), torch.from_numpy(y_mat).float(), torch.from_numpy(r_mat).float()
        
x_d = 32
y_d = 32
x, y, r = get_coordinates(x_d, y_d, batch_size=BATCH_SIZE)
if use_cuda:
    x = x.cuda()
    y = y.cuda()
    r = r.cuda()
    
class Generator(nn.Module):
    def __init__(self, x_dim, y_dim, batch_size=1, z_dim = 32, c_dim = 3, scale = 8.0, net_size = 128):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.net_size = net_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.scale = scale
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.scale = scale
        
        #Build NN graph
        self.linear1 = nn.Linear(z_dim, self.net_size)
        self.linear2 = nn.Linear(1, self.net_size, bias=False)
        self.linear3 = nn.Linear(1, self.net_size, bias=False)
        self.linear4 = nn.Linear(1, self.net_size, bias=False)
        
        self.linear5 = nn.Linear(self.net_size, self.net_size)
        self.linear6 = nn.Linear(self.net_size, self.net_size)
        self.linear7 = nn.Linear(self.net_size, self.net_size)
        self.linear8 = nn.Linear(self.net_size, self.c_dim)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.lin_seq = nn.Sequential(self.tanh, self.linear5, self.tanh, self.linear6, self.tanh,
                                 self.linear7, self.tanh, self.linear8, self.sigmoid)
        
    def forward(self, x, y, r, z):

        # self.scale * z?
        U = self.linear1(z) + self.linear2(x) + self.linear3(y) + self.linear4(r)   

        result = torch.transpose(self.lin_seq(U), 1, 2)
        
        return result.view(-1, 3, math.sqrt(result.size()[2]), math.sqrt(result.size()[2]))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

netG = Generator(x_dim = x_d, y_dim = y_d, z_dim=LATENT_DIM, batch_size = BATCH_SIZE)
netD = Discriminator()
print (netG)
print (netD)

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# For generating samples
def generate_image(frame, netG):
    
    noise = torch.randn(BATCH_SIZE, LATENT_DIM)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise, volatile=True)
    
    ones = torch.ones(BATCH_SIZE, 32 * 32, 1)
    if use_cuda:
        ones = ones.cuda()
        
    seed = torch.bmm(ones, noisev.unsqueeze(1))
    
    samples = netG(x, y, r, seed)
    
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, './tmp/cifar10/samples_{}.jpg'.format(frame))

# For calculating inception score
def get_inception_score(G, ):
    all_samples = []
    for i in range(10):
        samples_100 = torch.randn(100, 128)
        if use_cuda:
            samples_100 = samples_100.cuda(gpu)
        samples_100 = autograd.Variable(samples_100, volatile=True)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterator
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)

def inf_train_gen():
    while True:
        for images in train_gen():
            yield images
            
gen = inf_train_gen()
preprocess = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
if __name__ == "__main__":
    for iteration in range(ITERS):
        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for i in range(CRITIC_ITERS):
            _data = next(gen)
            netD.zero_grad()
    
            # train with real
            _data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            real_data = torch.stack([preprocess(item) for item in _data])
    
            if use_cuda:
                real_data = real_data.cuda(gpu)
            real_data_v = autograd.Variable(real_data)
    
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)
    
            # train with fake
            noise = torch.randn(BATCH_SIZE, 128)
            if use_cuda:
                noise = noise.cuda(gpu)
                
            with torch.no_grad():
                noisev= autograd.Variable(noise)
                        
            ones = torch.ones(BATCH_SIZE, 32 * 32, 1)
            if use_cuda:
                ones = ones.cuda()
                        
            seed = torch.bmm(ones, noisev.unsqueeze(1))
            
            fake = autograd.Variable(netG(x, y, r, seed).data)
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)
    
            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()
    
            # print "gradien_penalty: ", gradient_penalty
    
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()
        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
    
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
                
        with torch.no_grad():
            noisev= autograd.Variable(noise)
                        
        ones = torch.ones(BATCH_SIZE, 32 * 32, 1)
        if use_cuda:
            ones = ones.cuda()
                        
        seed = torch.bmm(ones, noisev.unsqueeze(1))
        fake = netG(x, y, r, seed)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()
    
        # Write logs and save samples
        lib.plot.plot('./tmp/cifar10/train disc cost', D_cost.cpu().data.numpy())
        lib.plot.plot('./tmp/cifar10/time', time.time() - start_time)
        lib.plot.plot('./tmp/cifar10/train gen cost', G_cost.cpu().data.numpy())
        lib.plot.plot('./tmp/cifar10/wasserstein distance', Wasserstein_D.cpu().data.numpy())
    
        # Calculate dev loss and generate samples every 100 iters
        if iteration % 200 == 99:
            dev_disc_costs = []
            for images in dev_gen():
                images = images.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
                imgs = torch.stack([preprocess(item) for item in images])
    
                # imgs = preprocess(images)
                if use_cuda:
                    imgs = imgs.cuda(gpu)
                imgs_v = autograd.Variable(imgs, volatile=True)
    
                D = netD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('./tmp/cifar10/dev disc cost', np.mean(dev_disc_costs))
    
            generate_image(iteration, netG)
        
        if iteration % 20000 == 0:
            torch.save(netG.state_dict(), "tmp/cifar10/G-cppn-wgan-cifar_{}.pth".format(iteration))
            torch.save(netD.state_dict(), "tmp/cifar10/D-cppn-wgan-cifar_{}.pth".format(iteration))
        
    
        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()
        lib.plot.tick()
