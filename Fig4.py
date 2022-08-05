'''
Plots growth rate as a function of resistivity for a variety of simulations
'''
import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('apj.mplstyle')
fig = plt.figure(figsize=(3.3,2.04))
N21_dark = "#7570B3"
N21_light = "#8DA0CB"
PC_dark = "#D95F02"
PC_light = '#FC8D62'
POT_dark = "#1B9E77"
POT_light = '#66C2A5'
m0_dark = "#CCBB44"
m0_light = "#DDCC77"
C = 1.3
ax = fig.add_subplot(1,1,1)
#N21
nun2 = C*np.array([1e-3,1e-4,1e-5])
gamman2 = C*np.array([0.103,0.040,0.0125])
pn2 = np.polyfit(np.log(nun2),np.log(gamman2),1)
pren2 = float(np.exp(pn2[1]))
print("N21: exp, pre ")
print(pn2[0])
print(pren2)
#plt.plot(nun2,nun2**pn2[0]*pren2,color=N21_dark)
#plt.scatter(nun2,gamman2,color=N21_light,edgecolors=N21_dark,zorder=100,linewidths=0.5,s=40,label
#="N21")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\eta \,\, \left(\frac{1}{v_a R}\right)$ ')
plt.ylabel(r'$\gamma \,\, \left(\frac{R}{v_a}\right)$')
plt.text(x=0.40,y=0.21,s=r"2.9$\eta^{0.458}$",transform=ax.transAxes,rotation=30, color=N21_dark)

#PC
nuPC = C*np.array([1e-3,1e-4,1e-5,1e-6])
gammaPC = C*np.array([0.11,0.0475,0.0168,0.00616])
pPC = np.polyfit(np.log(nuPC),np.log(gammaPC),1)
prePC = float(np.exp(pPC[1]))
print("PC: exp, pre")
print(pPC[0])
print(prePC)
#POT
nu = C*np.array([1e-3,1e-4,1e-5,1e-6])
gamma = C*np.array([0.163,0.0725,0.0332,0.0161])
p = np.polyfit(np.log(nu),np.log(gamma),1)
pre = float(np.exp(p[1]))
print("POT: exp, pre")
print(p[0])
print(pre)

#m0
num0 = C*np.array([1e-3,1e-4,1e-5])
gammam0 = C*np.array([0.0958,0.0633,0.0325])
pm0 = np.polyfit(np.log(num0),np.log(gammam0),1)
prem0 = float(np.exp(pm0[1]))
print("m0: exp, pre")
print(pm0[0])
print(prem0)
#m0
plt.plot(num0,num0**pm0[0]*prem0,color=m0_dark)
plt.scatter(num0,gammam0,color=m0_light,edgecolors=m0_dark,zorder=100,linewidths=0.5,s=40,label="m0")
plt.text(x=0.3,y=0.55,s=r"0.6$\eta^{0.23}$",transform=ax.transAxes,rotation=17, color=m0_dark)
#POT
plt.plot(nu,nu**p[0]*pre,color=POT_dark)
plt.scatter(nu,gamma,color=POT_light,edgecolors=POT_dark,zorder=100,linewidths=0.5,s=40,label="POT")

#PC
plt.plot(nuPC,nuPC**pPC[0]*prePC,color=PC_dark)
plt.scatter(nuPC,gammaPC,color=PC_light,edgecolors=PC_dark,zorder=100,linewidths=0.5,s=40,label="PC")

#N21
plt.plot(nun2,nun2**pn2[0]*pren2,color=N21_dark)
plt.scatter(nun2,gamman2,color=N21_light,edgecolors=N21_dark,zorder=100,linewidths=0.5,s=40,label="$N^2$")

#PC
plt.text(x=0.13,y=0.18,s=r"2.5$\eta^{2/5}$",transform=ax.transAxes,rotation=30, color=PC_dark)
#POT
plt.text(x=0.07,y=0.36,s=r"1.9$\eta^{1/3}$",transform=ax.transAxes,rotation=25, color=POT_dark)



plt.legend(loc="lower right",frameon=False, labelspacing=0.4, columnspacing=0.4, handlelength=0.2)
plt.savefig("N21gammanu.pdf",dpi = 400, bbox_inches="tight")


