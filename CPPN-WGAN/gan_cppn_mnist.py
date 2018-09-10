import os, sys
sys.path.append(os.getcwd())

import time
import math
import matplotlib
matplotlib.use('Agg')
import numpy as np

import tflib as lib
from tflib import save_images
import tflib.mnist
import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
    
if not os.path.exists("tmp"):
    os.makedirs("tmp")

if not os.path.exists("tmp/mnist"):
    os.makedirs("tmp/mnist")

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 60000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
LATENT_DIM = 128 #dimension of latent variable sample z
    
#lib.print_model_settings(locals().copy())

# ==================CPPN Modifications======================

def get_coordinates(x_dim = 28, y_dim = 28, scale = 8, batch_size = 1):

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
        
x_d = 28
y_d = 28
x, y, r = get_coordinates(x_d, y_d, batch_size=BATCH_SIZE)
if use_cuda:
    x = x.cuda()
    y = y.cuda()
    r = r.cuda()

class Generator(nn.Module):
    def __init__(self, x_dim, y_dim, batch_size=1, z_dim = 32, c_dim = 1, scale = 8.0, net_size = 128, devid = -1,):
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
        
        #self.linear9 = nn.Linear(self.net_size, self.c_dim)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #self.softplus = nn.Softplus()
        
        self.lin_seq = nn.Sequential(self.tanh, self.linear5, self.tanh, self.linear6, self.tanh,
                                 self.linear7, self.tanh, self.linear8, self.sigmoid)
        
    def forward(self, x, y, r, z):

        # self.scale * z?
        U = self.linear1(z) + self.linear2(x) + self.linear3(y) + self.linear4(r)   
        
        result = self.lin_seq(U).squeeze(2).view(x.size()[0], math.sqrt(x.size()[1]), math.sqrt(x.size()[1])).unsqueeze(1)
        return result.view(-1, result.size()[2]*result.size()[3])

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)

def generate_image(frame, netG):
    noise = torch.randn(BATCH_SIZE, LATENT_DIM)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise, volatile=True)
    
    ones = torch.ones(BATCH_SIZE, 28 * 28, 1)
    if use_cuda:
        ones = ones.cuda()
            
    seed = torch.bmm(ones, noisev.unsqueeze(1))
    
    samples = netG(x, y, r, seed)
    samples = samples.view(BATCH_SIZE, 28, 28)
    # print samples.size()

    samples = samples.cpu().data.numpy()

    save_images.save_images(
        samples,
        'tmp/mnist/samples_{}.png'.format(frame)
    )

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
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

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================

if __name__ == "__main__":

    netG = Generator(x_dim = x_d, y_dim = y_d, z_dim=LATENT_DIM, batch_size = BATCH_SIZE)
    netD = Discriminator()
    print (netG)
    print (netD)
    
    if use_cuda:
        netD = netD.cuda(gpu)
        netG = netG.cuda(gpu)
    
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    one = torch.FloatTensor([1])
    mone = one * -1
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)
    
    data = inf_train_gen()
    
    for iteration in range(ITERS):
        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
    
        for iter_d in range(CRITIC_ITERS):
            
            _data = next(data)
            real_data = torch.Tensor(_data)
            if real_data.size()[0] == BATCH_SIZE:
    
                if use_cuda:
                    real_data = real_data.cuda(gpu)
                real_data_v = autograd.Variable(real_data)
        
                netD.zero_grad()
        
                # train with real
                D_real = netD(real_data_v)
                D_real = D_real.mean()
                # print D_real
                D_real.backward(mone)
        
                # train with fake
                noise = torch.randn(BATCH_SIZE, LATENT_DIM)
                if use_cuda:
                    noise = noise.cuda(gpu)

                with torch.no_grad():
                    noisev= autograd.Variable(noise)
                    
                ones = torch.ones(BATCH_SIZE, 28 * 28, 1)
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
        
                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                optimizerD.step()
    
        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
    
        noise = torch.randn(BATCH_SIZE, LATENT_DIM)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise)
        
        ones = torch.ones(BATCH_SIZE, 28 * 28, 1)
        if use_cuda:
            ones = ones.cuda()
                
        seed = torch.bmm(ones, noisev.unsqueeze(1))
        
        fake = netG(x, y, r, seed)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()
    
        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                imgs = torch.Tensor(images)
                if use_cuda:
                    imgs = imgs.cuda(gpu)
                imgs_v = autograd.Variable(imgs, volatile=True)
    
                D = netD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
    
            generate_image(iteration, netG)
        
        if iteration % 20000 == 0:
            torch.save(netG.state_dict(), "tmp/mnist/G-cppn-wgan_{}.pth".format(iteration))
            torch.save(netD.state_dict(), "tmp/mnist/D-cppn-wgan_{}.pth".format(iteration))
    
