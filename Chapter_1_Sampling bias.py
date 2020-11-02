#以下代码有黄酬子佑同学提供。
import numpy as np
import random
import matplotlib.pyplot as plt
N = 10000 # Population size  
np.random.seed(0)
x = np.random.uniform(1.0, 10.0, N) # Auxiliary information
err = np.random.normal(loc = 0, scale = 1, size = N)
y = (2 * x + 3 + err).tolist() # Targe
mu = np.mean(y)
prob = (x / sum(x)).tolist()  # Inclusion probability.

def one_sample_srs(n):  # Estimators under SRS.
    sample_i = random.sample(y, n)
    mean_i = np.mean(sample_i)  # Sample mean
    sd_i = np.std(sample_i) / np.sqrt(n)  # Sample standard error
    sd_modi_i = sd_i * np.sqrt(1 - n / N)  # Modified sample standard error
    return([(mean_i - mu) / sd_i, (mean_i - mu) / sd_modi_i])
    
def one_sample_poi(n): # Estimators under Poisson sampling.
    def poi_pi(prob):
        return np.random.binomial(1, prob * n)
    chosen = list(map(poi_pi, prob))
    sample_poi = [i for i, j in zip(y, chosen) if j == 1] # Sample index set $A$
    mean_i = np.mean(sample_poi) # Sample mean
    sd_i = np.std(sample_poi)/np.sqrt(n) # Sample standard error
    return (mean_i-mu)/sd_i

for n in [500,5000]:
    result_poi = []
    for i in range(1000):
        result_poi.append(one_sample_poi(n))
    plt.hist(result_poi) # Histogram under Poisson sampling
    plt.savefig('Figures\Selection_Bias_Poisson_' + str(n) + ".pdf") # Save fig
    plt.close()

for n in [500,5000]:
    result_srs = np.zeros([1000,2])
    for i in range(1000):
        result_srs[i,:] = one_sample_srs(n)
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (10,5)) # Two subplots
    ax1.hist(result_srs[:,0],range = (-4,4)) # The first subplot shows the histogram using traditional SE
    ax2.hist(result_srs[:,1],range = (-4,4)) # The second subplot shows the histogram using modified SE
    fig.savefig('Figures\Selection_Bias_SRS_'+str(n)+".pdf") #Save fig
    plt.close(fig)

    
