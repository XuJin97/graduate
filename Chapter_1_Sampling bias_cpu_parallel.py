import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
import multiprocessing
import time
from functools import partial


def one_sample_srs(n,x,y,prob,mu,N):  # Estimators under SRS.
    sample_index = np.random.choice(N, n, replace=False)  # Sample index set $A$
    sample_i = y[sample_index]
    mean_i = np.mean(sample_i)  # Sample mean
    sd_i = np.std(sample_i) / np.sqrt(n)  # Sample standard error
    sd_modi_i = sd_i * np.sqrt(1 - n / N)  # Modified sample standard error
    return([(mean_i - mu) / sd_i, (mean_i - mu) / sd_modi_i])


def one_sample_poi(n,x,y,prob,mu,N):  # Estimators under Poisson sampling.
    sample_poi = np.random.binomial(1, prob * n, [N, 1])  # Sample index set $A$
    sample_i = y[sample_poi[:, 0] == 1]
    mean_i = np.mean(sample_i)  # Sample mean
    sd_i = np.std(sample_i) / np.sqrt(n)  # Sample standard error
    return (mean_i - mu) / sd_i

def saveResult(result):## This is the call back function
    result_1.append(result)
    
if __name__ == '__main__':  # This one is needed for multiprocessing
    n_core = 15 # Number of cores we are using
    rs = check_random_state(12345)
    N = 10000  # Population size
    x = rs.uniform(1.0, 10.0, size=(N, 1))  # Auxiliary information
    y = 2 * x + 3 + rs.normal(loc=0.0, scale=1, size=(N))  # Target
    mu = np.mean(y)
    prob = x / sum(x)  # Inclusion probability.
    # for n in [500, 5000]:
    for n in [500,5000]:
        ###################################################################################
        ### SRS.
        begin_time = time.time()
        result_1 = []
        arg_rep = [n]*1000
        func = partial(one_sample_srs,x=x,y=y,prob=prob,mu=mu,N=N)
        pool = multiprocessing.Pool(processes=n_core) # This is the number of threads (CPUs) for parallel 
        pool.map_async(func,arg_rep,callback = saveResult)
        pool.close()  
        pool.join()  

        result_srs = np.array(result_1[0])
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (10,5))  # Two subplots
        fig.suptitle('Simple Random Sampling with SS '+str(n)) # Title
        ax1.hist(result_srs[:,0],range = (-4,4))  # The first subplot shows the histogram using traditional SE
        ax2.hist(result_srs[:,1],range = (-4,4)) # The second subplot shows the histogram using modified SE
        fig.savefig('Figures\Selection_Bias_SRS_'+str(n)+"_parallel.pdf") #Save fig
        plt.close(fig)
        time_para = round(time.time()-begin_time,2)
        print("It takes about " + str(time_para) + " seconds using parallel for SRS with SS " + str(n))
        
        begin_time = time.time()
        result_srs = np.zeros([1000,2])
        for i in range(1000):
            result_srs[i,:] = np.array(one_sample_srs(n, x,y, prob,mu,N))
        
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (10,5))  # Two subplots
        fig.suptitle('Simple Random Sampling with SS '+str(n)) # Title
        ax1.hist(result_srs[:,0],range = (-4,4))  # The first subplot shows the histogram using traditional SE
        ax2.hist(result_srs[:,1],range = (-4,4)) # The second subplot shows the histogram using modified SE
        fig.savefig('Figures\Selection_Bias_SRS_'+str(n)+".pdf") #Save fig
        plt.close(fig)
        time_single = round(time.time()- begin_time,2)
        print("It takes about " + str(time_single) + " seconds WITHOUT using parallel for SRS with SS " + str(n)) 
        
        ###################################################################################
        # ### Poisson sampling.
        begin_time = time.time()
        result_1 = []
        arg_rep = [n]*1000
        func = partial(one_sample_poi,x=x,y=y,prob=prob,mu=mu,N=N)
        pool = multiprocessing.Pool(processes=n_core) # This is the number of threads (CPUs) for parallel 
         #srs_para = pool.map_async(one_sample_srs, arg_rep,callback = result_srs.extend)  #Parallel Processing in Python - A Practical Guide with Examples
        pool.map_async(func,arg_rep,callback = saveResult)
        pool.close()  
        pool.join()  
        
        result_poi = np.array(result_1[0])
        plt.hist(result_poi)  # Histogram under Poisson sampling
        plt.title("Poisson Sampling with SS "+ str(n)) # Title
        plt.savefig('Figures\Selection_Bias_Poisson_'+str(n)+"_parallel.pdf") # Save fig
        plt.close()
        time_para = round(time.time()-begin_time,2)
        print("It takes about " + str(time_para) + " seconds using parallel for Poisson sampling with SS " + str(n))
        
        begin_time = time.time()
        result_poi = np.zeros([1000,1])
        for i in range(1000):
            result_poi[i,:] = one_sample_poi(n, x,y, prob,mu,N)
        
        plt.hist(result_poi)  # Histogram under Poisson sampling
        plt.title("Poisson Sampling with SS "+ str(n)) # Title
        plt.savefig('Figures\Selection_Bias_Poisson_'+str(n)+".pdf") # Save fig
        plt.close()
        time_single = round(time.time()- begin_time,2)
        print("It takes about " + str(time_single) + " seconds WITHOUT using parallel for Poisson sampling with SS " + str(n))       
        
        
        