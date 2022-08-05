'''
Plots u_r^2/u^2 and B_r^2/B_r as a function of N_0^2
'''

import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('apj.mplstyle')
fig = plt.figure(figsize=(3.3,4.08))
lg = "#b2abd2"
pi = "#e66101"
color = gr = "#5e3c99"
lp = '#fdb863'
C = 1.3*1.3
#ax1 = fig.add_subplot(2,1,1)
#ax2 = fig.add_subplot(2,1,2)
ax2 = fig.add_axes([0,0.5,1,0.5])
ax1 = fig.add_axes([0,0,1,0.5])

n2 = C*np.array([1, 0.1, 0.2,  0.4, 0.6, 100])
rat = np.array([0.189,0.18868867, 0.18923287, 0.18928875, 0.18927571, 0.18876571])
n2small = C*np.array([0.01,0.001,0.02, 0.05])
ratsmall = np.array([0.229,0.229,0.2222,0.197468])


ax1.scatter(n2small,ratsmall,color=lp,edgecolors=pi,zorder=100,linewidths=0.5,s=40)
ax1.scatter(n2,rat,color=lg,edgecolors=gr,zorder=100,linewidths=0.5,s=40)

ax1.set_xscale('log')

ax1.set_xlabel(r'$N_0^2$')
ax1.set_ylim(0,0.27)
ax1.set_xlim(7e-4,2e2)
ax1.set_ylabel(r'$\frac{ \langle B_r^2 \rangle }{\langle B^2 \rangle }$')
#ax1.set_yticks([0.18,0.2,0.22])

n2_2 = C*np.array([ 0, 1,  0.2, 100,0.4,0.6,0.1,10])
rat = np.array([ 0.0855,5.684E-6, 8.317E-5, 3.145E-7,2.2657e-5,1.1504e-5,0.000294158, 7.53217e-7])
n2small = C*np.array([0.001,0.02, 0.01, 0.05])
ratsmall = np.array([0.0662,0.00492,0.0146, 0.0007323])

ax2.scatter(n2_2,rat,color=lg,edgecolors=gr,zorder=100,linewidths=0.5,s=40,label="simulations")
ax2.scatter(n2small,ratsmall, color=lp, edgecolors=pi, zorder=100, s=40, linewidths=0.5)
ax2.set_xscale('log')
ax2.set_xlim(7e-4,2e2)
ax2.set_yscale('log')
ax2.axhline(0.0855, color=pi, linestyle='--')
ax1.axhline(0.2284, color=pi, linestyle='--')
ax2.set_xticks([])
#ax2.set_xlabel(r'$N^2$')
ax2.set_ylabel(r'$\frac{\langle u_r^2 \rangle }{ \langle u^2 \rangle}$')
ax2.set_yticks([1e-7,1E-5,1e-3,1e-1])
plt.savefig("n2comb.pdf",dpi = 400, bbox_inches="tight")

