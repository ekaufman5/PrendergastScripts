'''
Plots the RHS of eqn 12 as a function of N_0^2
'''

import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('apj.mplstyle')
fig = plt.figure(figsize=(3.3,2.04))
lg = "#b8e186"
pi = "#d01c8b"
color = gr = "#4dac26"
lp = '#f1b6da'
C = 1.3*1.3

#create data arrays 
n2 = C*np.array([0.001, 0.02,0.01, 0.05])
n2_1big = C*np.array([0.1,0.2,0.6,0.4,1,100])
mag = np.array([0.00790976, 0.00487563, 0.00615076, 0.00289101])
magbig = np.array([0.00247951,0.0021928, 0.002476, 0.002482,0.00247599,0.00246308])

#create plot
plt.scatter(n2,mag,color=pi)
plt.scatter(n2_1big,magbig,color=gr)
plt.xscale('log')
plt.xlabel("N^2")
plt.ylabel("<er curl(uxB_0)B_r>/B^2")
plt.savefig("utheta.pdf",dpi = 400, bbox_inches="tight")
