import torch.nn as nn
import numpy as np
import torch
import math

def get_coordinates(x_dim = 28, y_dim = 28, scale = 1, batch_size = 1):
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
         

class DC_Discriminator(nn.Module):
    def __init__(self,nc,ndf):
        super(DC_Discriminator,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=2),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=0, bias=True),
                                nn.Sigmoid())

    def forward(self,x):
        
        out = self.layer1(x)      
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out.squeeze(1).squeeze(1)

class DC_Generator(nn.Module):
    def __init__(self, x_dim, y_dim, batch_size=1, z_dim = 32, c_dim = 1, scale = 1.0, net_size = 128, devid = -1):
        super(DC_Generator, self).__init__()
        self.batch_size = batch_size
        self.net_size = net_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.scale = scale
        self.c_dim = c_dim
        self.z_dim = z_dim
        
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
        
        
        lin_list = [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5, self.linear6, 
                    self.linear7, self.linear8]
        
        for layer in lin_list:
            layer.weight.data.normal_(0, 1)
            try:
                layer.bias.data.fill_(0)
            except:
                pass
            
    
    def forward(self, x, y, r, z):

        U = self.linear1(z) + self.linear2(x) + self.linear3(y) + self.linear4(r)   
        
        result = self.lin_seq(U).squeeze(2).view(x.size()[0], round(math.sqrt(x.size()[1])), round(math.sqrt(x.size()[1]))).unsqueeze(1)
        
        return result
    
    
    
    
#class DC_Generator(nn.Module):
#    def __init__(self, nc, ngf, nz):
#        super(DC_Generator,self).__init__()
#        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz,ngf*4,kernel_size=3),
#                                 nn.BatchNorm2d(ngf*4),
#                                 nn.ReLU())
#        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=3,stride=2,padding=0),
#                                 nn.BatchNorm2d(ngf*2),
#                                 nn.ReLU())
#        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1),
#                                 nn.BatchNorm2d(ngf),
#                                 nn.ReLU())
#        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1),
#                                 nn.Tanh())
#
#    def forward(self,x):
#        
#        print(x.size())
#        raise Exception() 
#
#        out = self.layer1(x)      
#        out = self.layer2(out)   
#        out = self.layer3(out)
#        out = self.layer4(out)
#        
#        return out