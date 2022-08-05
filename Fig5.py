'''
Plots growth rate normalized to the growth rate at N_0^2=0 as a function of N_0^2
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apj.mplstyle')
fig = plt.figure(figsize=(3.3,2.04))
lg = "#b2abd2"
pi = "#e66101"
color = gr = "#5e3c99"
lp = '#fdb863'
C = 1.3*1.3
ax = fig.add_subplot(1,1,1)
n2 = C*np.array([0.001, 0.02,0.01, 0.05])
n2_1big = C*np.array([0.1,0.2,0.4,0.6,1,10,100])

gamma_0 = 0.0332
gamma = np.array([0.0323, 0.0213, 0.0263, 0.0128])
gamma_big = np.array([0.0127, 0.0126, 0.0126,0.0126, 0.0125,0.0125,0.0125])

gamma_norm = gamma/gamma_0
gamma_norm_big = gamma_big/gamma_0

plt.scatter(n2,gamma_norm,color=lp,edgecolors=pi,zorder=100,linewidths=0.5,s=40)
plt.scatter(n2_1big,gamma_norm_big,color=lg,edgecolors=gr,zorder=100,linewidths=0.5,s=40)

plt.plot([1e-4,1e3],[1,1],linestyle='--',color=pi)
plt.xscale('log')

ax.set_xlabel(r'$N_0^2$')
ax.set_ylim(0,1.05)
ax.set_xlim(7e-4,2e2)
ax.set_ylabel(r'$\frac{\gamma (N_0^2)}{\gamma (N_0^2 = 0)}$')
plt.savefig("gammanorm.pdf",dpi = 400, bbox_inches="tight")

