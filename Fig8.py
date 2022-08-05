'''
Plots B_x as a function of x for a range of resistivities
'''
import h5py
import matplotlib.pyplot as plt
import numpy as np
import dedalus.public as d3

plt.style.use('apj.mplstyle')
fig = plt.figure(figsize=(3.3,4.08))

c3_d = "#a6cee3"
c4_d = "#b2df8a"
c5_d = "#1f78b4"
c6_d = "#33a02c"

radius = 1
Lmax = 127
L_dealias = 3/2
N_phi = 4*64
Nmax = 127
N_dealias = 3/2
dealias_tuple = (1, 1, 4)
dtype = np.float64
mesh = None

# Bases
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor(c, dtype=dtype, mesh=mesh)
b = d3.BallBasis(c, shape=(N_phi, Lmax+1, Nmax+1), radius=radius, dealias=dealias_tuple, dtype=dtype) 
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids()

def barray(fi):
    B_vec = d.VectorField(c, bases=b, name='B')
    f = h5py.File(fi, mode='r')
    B_vec['g'] = np.array(f['tasks/B'][0])
    f.close()
    zero = B_vec(r=1e-5, theta=np.pi/2, phi=0).evaluate()['g']
    Bx_right = B_vec(theta=np.pi/2, phi=0).evaluate()['g']
    Bx_left = B_vec(theta=np.pi/2, phi=2*np.pi/2).evaluate()['g']

    zero = np.array(zero[2,0,0,0])
    Bx_right = np.array(Bx_right[2,0,0])
    Bx_left = np.array(Bx_left[2,0,0])
    Bx_left = np.flip(Bx_left)
    nleft = np.append(Bx_left*-1, zero)
    B = np.concatenate((nleft, Bx_right))
    return B, zero

B3, z3 = barray('diff-3/test_outputs/slices/slices_s42.h5') 
print("3")
B4, z4 = barray('diff-4/test_outputs/slices/slices_s78.h5') 
print("4")
B5, z5 = barray('diff-5/test_outputs/slices/slices_s23.h5')
print("5")

radius = 1
Lmax = 511
L_dealias = 3/2
N_phi = 4*64
Nmax = 511
N_dealias = 3/2
dealias_tuple = (1, L_dealias, N_dealias)
dtype = np.float64
mesh = None
# Bases
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor(c, dtype=dtype, mesh=mesh)
b = d3.BallBasis(c, shape=(N_phi, Lmax+1, Nmax+1), radius=radius, dtype=dtype)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids()
B_vec = d.VectorField(c, bases=b, name='B')
f = h5py.File('../../511_res_incompressible/PCBC/test_outputs/slices/slices_s5.h5', mode='r')
B_vec['g'] = np.array(f['tasks/B'][0])
rB = f['tasks/B'].dims[3][0][:].ravel()
f.close()

zero = B_vec(r=1e-5, theta=np.pi/2, phi=0).evaluate()['g']
Bx_right = B_vec(theta=np.pi/2, phi=0).evaluate()['g']
Bx_left = B_vec(theta=np.pi/2, phi=2*np.pi/2).evaluate()['g']
zero = np.array(zero[2,0,0,0])
Bx_right = np.array(Bx_right[2,0,0])
Bx_left = np.array(Bx_left[2,0,0])

Bx_left = np.flip(Bx_left)
nleft = np.append(Bx_left*-1, zero)
B = np.concatenate((nleft, Bx_right))

rneg = np.array(rB*-1)
rneg = np.flip(rneg)
rneg = np.append(rneg, 1e-5)
rplot = np.concatenate((rneg, rB))

ax1 = fig.add_axes([0,0.55,1,0.43])
ax2 = fig.add_axes([0,0,1,0.43])
xlim = 0.07

ax1.axvline(xlim, color='#bababa', linestyle='--')
ax1.axvline(-xlim, color='#bababa', linestyle='--')


ax1.set_xlabel(r"$x$")
ax2.set_xlabel(r"$x$")
ax1.set_ylabel(r'$B_x$')
ax2.set_ylabel(r'$B_x$')
ax2.set_ylim(0.98,1.003)
ax2.set_xlim(-xlim,xlim)


ax1.plot(rplot,B3/z3, label=r"D3",color=c3_d)
ax2.plot(rplot,B3/z3, label=r"D3$",color=c3_d)

ax1.plot(rplot,B4/z4, label="D4",color=c4_d)
ax2.plot(rplot,B4/z4, label=r"D4",color=c4_d)

ax1.plot(rplot,B5/z5, label=r"D5",color=c5_d)
ax2.plot(rplot,B5/z5, label=r"D5",color=c5_d)

ax1.plot(rplot,B/zero, label=r"D6",color=c6_d)
ax2.plot(rplot, B/zero, label=r"D6",color=c6_d)

ax1.legend(frameon=False, loc='upper right', labelspacing=0.4, columnspacing=0.4, handlelength=0.8)
plt.savefig("BxDiff.pdf",dpi=400,bbox_inches="tight")
