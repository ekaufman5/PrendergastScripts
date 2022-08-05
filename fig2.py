"""
Plots KE as a function of time 
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt 
from collections import OrderedDict
plt.style.use('apj.mplstyle')
fig = plt.figure(figsize=(3.3,2.04))
lg = "#b2abd2"
gr = "#5e3c99" 
pi = "#e66101"
lp = '#fdb863'

ax = fig.add_subplot(1,1,1)
f = h5py.File('test_outputs/scalar/scalar_s1/scalar_s1_p0.h5')
ke = np.array(f['tasks/KE'])[:,0,0,0]
time = np.array(f['scales/sim_time'])
time = time/1.3
i = np.argmin(np.abs(time-300))
i2 = np.argmin(np.abs(time-2000))
p = np.polyfit(time[i:i2],np.log(ke[i:i2]),1)
poly = np.poly1d(p)

tlong = np.linspace(-200,2000,1000)
linestyles = OrderedDict(
    [

     ('loosely dashed',      (0, (1, 2))),
     ('dashed',              (0, (10, 4))),
     ('densely dashed',      (0, (1, 3))),

    ])
lw = 4
plt.clf()
fig, ax = plt.subplots()

ax.semilogy(time,ke, color=gr, label='simulation', linewidth=lw)
ax.plot(tlong,np.exp(poly(tlong)), color=pi, linewidth=1)

plt.xlabel('$t$')
ax.set_ylabel('$KE$')
fig.text(x=0.15,y=0.25,s=r"$e^{%s t}$"%float('%.3g' % p[0]),transform=ax.transAxes,rotation=35, color=pi, fontsize=25)
plt.xlim(-50,1650)
plt.ylim(0.00001,1e26)
plt.savefig("ke.pdf",dpi = 400, bbox_inches="tight")

