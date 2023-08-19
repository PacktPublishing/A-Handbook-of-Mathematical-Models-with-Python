
import numpy as np
import math, random
import matplotlib.pyplot as plt

current_vel, current_disp, current_accel = 2, 0, 0
total_time = 100
accel_dict = {0:0,5:2,10:8, 20: -2,40:5,45: 9, 60: -3,85:0}

true_values = []
for t in range(1, total_time+1):
    
    current_disp = current_disp + current_vel + (1/2)*current_accel
    
    try:
        current_accel = accel_dict[t]
    except KeyError:
        pass
    
    current_vel = current_vel + current_accel
    true_values.append((current_disp, current_vel, current_accel))

err_range = [700, 30, 15]
measurements = []
for item in true_values:
    d,v,a = item
    
    random_err = [random.randint(-1*error_range[0], err_range[0]), random.randint(-1*err_range[1], err_range[1]), 
                  random.randint(-1*err_range[2], err_range[2])]
    
    new_disp = d + random_err[0] if d+random_err[0] >0 else 0
    new_vel = v + random_err[1]
    new_accel = a + random_err[2]
    measurements.append((new_disp, new_vel, new_accel))

plt.plot([i for i in range(total_time)], [y[0] for y in true_values], 'r--', label = 'True Values')
plt.plot([i for i in range(total_time)], [y[0] for y in measurements], 'b--', label = 'Measurements')
plt.ylabel("Displacement")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
