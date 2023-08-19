
import math, random
import numpy as np
import matplotlib.pyplot as plt

x_k = np.asarray([30,20]) 
Q = np.asarray([[0.004,0.002],[0.002,0.001]]) 
A = np.asarray([[1,1],[0,1]]) 
R = np.asarray([[0.4,0.01],[0.04,0.01]]) 
H = np.asarray([[1,0],[0,1]]) 
P = np.asarray([[0,0],[0,0]])  

estimation = []
for k_loop in range(total_time):
    
    z_k = np.asarray([measurements[k_loop][0], measurements[k_loop][1]])
    
    x_k = A.dot(x_k) 
    P = (A.dot(P)).dot(A.T) + Q 
    
    K = (P.dot(H.T)).dot(np.linalg.inv((H.dot(P).dot(H.T)) + R)) 
    x_k = x_k + K.dot((z_k - H.dot(x_k))) 
    
    P = (np.identity(2) - K.dot(H)).dot(P)
    estimation.append((x_k[0], x_k[1])) 

plt.plot([i for i in range(total_time)], [y[0] for y in true_values], 'r--', label = 'True Values')
plt.plot([i for i in range(total_time)], [y[0] for y in measurements], 'b--', label = 'Measurements')
plt.plot([i for i in range(total_time)], [y[0] for y in estimation], 'g--', label = 'Estimated Values')
plt.title('Estimation of displacement')
plt.ylabel("Displacement")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
