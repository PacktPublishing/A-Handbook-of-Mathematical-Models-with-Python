
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)

def gibbs_sampler(mus, sigmas, n_iter = 10000):
  samples = []
  y = mus[1]
  for _ in range(n_iter):
    x = p_x_y(y, mus, sigmas)
    y = p_y_x(x, mus, sigmas)
    samples.append([x, y])
  return samples
  
def p_x_y(y, mus, sigmas):
  mu = mus[0] + sigmas[1, 0]/sigmas[0, 0] * (y - mus[1])
  sigma = sigmas[0, 0]-sigmas[1, 0]/sigmas[1, 1]*sigmas[1, 0]
  return np.random.normal(mu, sigma)

def p_y_x(x, mus, sigmas):
  mu = mus[1] + sigmas[0, 1] / sigmas[1, 1]*(x - mus[0])
  sigma = sigmas[1, 1] - sigmas[0, 1]/sigmas[0, 0]*sigmas[0, 1]
  return np.random.normal(mu, sigma)

mus = np.asarray([5, 5])
sigmas = np.asarray([[1, 0.9], [0.9, 1]])
samples = gibbs_sampler(mus, sigmas)
burnin = 200
x = list(zip(*samples[burnin:]))[0]
y = list(zip(*samples[burnin:]))[1]

sns.jointplot(samples[burnin:], x = x, y = y, kind = 'kde')
sns.jointplot(samples[burnin:], x = x, y = y, kind = 'reg')
plt.show()
