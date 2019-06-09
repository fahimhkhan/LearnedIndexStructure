import time
import os
import random
import math
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from bisect import bisect_left
import matplotlib.pyplot as plt


#Functions for dataset Generation
#################################

def generate_dataset(N):
    dataset = [exponential() for _ in range(N)]
    #dataset = [lognormal() for _ in range(N)]
    #dataset = [laplace() for _ in range(N)]
    #dataset = [uniform() for _ in range(N)]
    #dataset = [gaussian() for _ in range(N)]
    dataset = sorted(dataset)
    def KVrand():
        x = random.choice(dataset)
        y = dataset.index(x)
        return x, y
    return dataset, KVrand

def exponential(lambda1=1.0):
    u = random.random()
    x = - math.log(u) / lambda1
    return x 

def lognormal(mu=0, sigma=5.0):
    x = random.lognormvariate(mu, sigma)
    return x
  
def uniform(a=0.0,b=10.0):
    x = random.uniform(a,b)
    return x

def gaussian(mu=0, sigma=5.0):
    x = random.gauss(mu, sigma)
    return x

def laplace():
    u = random.random()
    if(u<=0.5):
        x = math.log(2*u)
    else:
        x = -math.log(2-2*u)    
    return x


#Functions for Neural Network Configuration
###########################################

def create_NN(dim=128):
    NN = torch.nn.Sequential(torch.nn.Linear(1, dim),torch.nn.ReLU(),torch.nn.Linear(dim, 1),)
    return NN


def to_tensor(x):   
    return torch.unsqueeze(Variable(torch.Tensor(x)), 1)


#Traditional Search Functions
#############################

def linear_search(x, dataset):
    for idx, n in enumerate(dataset):
        if n > x:
            break
    return idx - 1


def binary_search(x, dataset):
    i = bisect_left(dataset, x)
    if i:
        return i - 1
    raise ValueError

#Main
#####

def main():
    N = 1000
    lr = 0.0001
    batch_no = 0
    LF_x = []
    LF_y = []  
    minloss = N
    
    
    dataset, KVrand = generate_dataset(N)
    NN = create_NN()
    optimizer = torch.optim.Adam(NN.parameters(), lr=lr)
    
#Training
#########

    start = time.time()
    try:
        while True:
            batch_no = batch_no + 1
            batch_x = []; batch_y = []
            for _ in range(256):
                x, y = KVrand()
                batch_x.append(x)
                batch_y.append(y)

            batch_x = to_tensor(batch_x)
            batch_y = to_tensor(batch_y)

            Predicted_idx = NN(batch_x) * N

            output = F.smooth_l1_loss(Predicted_idx, batch_y)
            loss = output.data
                      
            if (minloss>loss.item()):
               minloss=loss.item()
               print('Minloss =',minloss,'at',time.time())

            #print(loss, minloss,'at',time.time())  
            
            LF_x.append(batch_no)
            LF_y.append(loss)
            
            if (loss.item()<1.0):
                break
            
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
    except KeyboardInterrupt:
        pass
    end = time.time()
    TrainingTime = end - start
    print('Time required for Training =', TrainingTime)

#Convergence Plot
#################
    plt.plot(LF_x, LF_y, 'b', label='Learned Index Structure')
    plt.xlabel('Number of Batches')
    plt.ylabel('Loss')
    plt.title('Learning Convergence')
    plt.legend(loc='best')
    plt.show()

    #import pdb
    #pdb.set_trace()

if __name__ == '__main__':
    main()
