
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

def density(z):
  z = np.reshape(z, [z.shape[0], 2])
  z1, z2 = z[:, 0], z[:, 1]
  norm = np.sqrt(z1 ** 2 + z2 ** 2)
  exp1 = np.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
  exp2 = np.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
  v = 0.5 * ((norm - 2) / 0.4) ** 2 - np.log(exp1 + exp2)
  return np.exp(-v)\

r = np.linspace(-5, 5, 1000)
z = np.array(np.meshgrid(r, r)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

def metropolis_sampler(target_density, size = 100000):
  burnin = 5000
  size += burnin
  x0 = np.array([[0, 0]])
  xt = x0
  samples = []
  for i in tqdm(range(size)):
    xt_candidate = np.array([np.random.multivariate_normal(xt[0], np.eye(2))])
    accept_prob = (target_density(xt_candidate))/(target_density(xt))
    if np.random.uniform(0, 1) < accept_prob:
      xt = xt_candidate
    samples.append(xt)
  samples = np.array(samples[burnin:])
  samples = np.reshape(samples, [samples.shape[0], 2])
  return samples

q = density(z) #true density
plt.hexbin(z[:,0], z[:,1], C = q.squeeze())
plt.gca().set_aspect('equal', adjustable ='box')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()

samples = metropolis_sampler(density)
plt.hexbin(samples[:,0], samples[:,1])
plt.gca().set_aspect('equal', adjustable = 'box')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()


#samples = np.random.multivariate_normal(mus, sigmas, 10000)
#x = list(zip(*samples[burnin:]))[0]
#y = list(zip(*samples[burnin:]))[1]
#burnin = 200
#sns.jointplot(samples[burnin:], x = x, y = y, kind = 'kde')
#plt.show()
