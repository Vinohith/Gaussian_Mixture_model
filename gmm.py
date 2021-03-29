# importing some basic libraries for numerical computations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



# Definition of the 1_D gaussian distribution
def gaussian_def(data, mu_k, var_k):
  diff = (data - mu_k).T
  return (1. / (2.*np.pi*var_k)**(0.5)) * np.exp(-0.5*((diff**2)*(1./var_k)))



# Definition of the Expectation Step
def E_step(data, clusters):
  # For each datapoint, denominator stores the sum of the probabilities of the point  
  # belonging to a each of the clusters.
  # i.e. it stores the value of the denominator in the gamma equation for each datapoint.
  denominator = np.zeros((data.size, 1)) # (N, 1) vector
  # loop over each datapoint
  for i, datapoint in enumerate(data):
    # for each cluster calculate the numerator and update the denominator (summation
    # of the numerators calculated over all the clusters)
    for cluster in clusters:
      pi_k = cluster['pi_k']
      mu_k = cluster['mu_k']
      var_k = cluster['var_k']
      numerator = pi_k*gaussian_def(datapoint, mu_k, var_k)[0]
      cluster['gamma_nk'][i] = numerator
      # the denominator acts as a normalization. So, it the summation of all the 
      # values of the numerator for the datapoint
      denominator[i] += numerator
    # finally normalize using the denominator to get gamma(z_nk) for each datapoint
    # in each cluster
    for cluster in clusters:
      cluster["gamma_nk"][i] /= denominator[i]


# Definition of the Maximization Step
def M_step(data, clusters):
  N = data.size
  # loop over each cluster and update its parameters
  for cluster in clusters:
    gamma_nk = cluster['gamma_nk']
    # Calculate N_k
    N_k = np.sum(gamma_nk)
    # Update mean
    mu_k_new = np.sum(gamma_nk*data)/N_k
    # Update variance
    var_k_new = np.sum(gamma_nk*(data-mu_k_new)**2)/N_k
    # Update Mixing coellicient
    pi_k_new = N_k/N
    cluster['mu_k'] = mu_k_new
    cluster['var_k'] = var_k_new
    cluster['pi_k'] = pi_k_new



# Definig some synthetic 1-D data
np.random.seed(1)
data_1 = np.random.normal(5, 2, size=500)
data_2 = np.random.normal(15, 2, size=500)

data = np.concatenate((data_1,data_2)).reshape(-1, 1)
print("Number of datapoints : ", data.size)

plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
plt.show()

# initialize the clusters
n_clusters = 2
clusters = []
for i in range(n_clusters):
  clusters.append({'pi_k' : 1.0/n_clusters,
                   'mu_k' : i*10.0,
                   'var_k' : 1.0,
                   'gamma_nk' : np.zeros((data.size, 1))
                   })

# Fit the model to the data and find the mixture models parameters that best fit
# the data by alternating between the E and M steps
n_epochs = 10
for i in range(n_epochs):
  print("EPOCH ", i)
  E_step(data, clusters)
  M_step(data, clusters)
  # plotting the curves
  plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
  xmin, xmax = plt.xlim()
  x = np.linspace(xmin, xmax, 100)
  for cluster in clusters:
    print("pi = {}, mu = {}, var = {}".format(cluster['pi_k'], cluster['mu_k'], cluster['var_k']))
    p = norm.pdf(x, cluster['mu_k'], np.sqrt(cluster['var_k']))*cluster['pi_k']
    plt.plot(x, p, 'k', linewidth=2)
  # p1 = norm.pdf(x, )
  plt.show()
  print("********************************")