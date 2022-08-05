import numpy as np
import os 
import sys
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI
logger = logging.getLogger(__name__)
radius = 1 
Lmax = 127 
L_dealias = 3/2 
N_phi = 8 
Nmax = 127 
N_dealias = 3/2 
dealias_tuple = (1, L_dealias, N_dealias)
dtype = np.float64
mesh = [2,64] #None
omega = 2.
r_0 = 0.875
delta_r = 0.04
tau = 3.
r_top = 0.95

# Bases
c = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(c, dtype=dtype)
b = d3.BallBasis(c, shape=(N_phi, Lmax+1, Nmax+1), radius=radius, dealias=dealias_tuple, dtype=dtype)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, L_dealias, N_dealias))
m = 2
F_func = lambda t: np.sin(theta)**m*np.sin(m*phi)*np.exp(-(r-r_0)**2/delta_r**2)

u = dist.VectorField(c, bases=b, name='u')
u.set_scales(dealias_tuple)
tau_p = dist.Field(bases=b_S2, name='tau_p')
p = dist.Field(bases=b, name='p')
u['g'][2] = F_func(0)
LiftTau = lambda A: d3.LiftTau(A, b, -1)

output_dir = './test_outputs/'
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(output_dir)):
                os.makedirs('{:s}/'.format(output_dir))

# Problem
problem = d3.LBVP([p,tau_p], namespace=locals())
problem.add_equation("lap(p) + LiftTau(tau_p) = - d3.div(u)")
problem.add_equation("radial(grad(p)(r=1)) = 0", condition="ntheta != 0")
problem.add_equation("p(r=1) = 0", condition="ntheta == 0")

gp = d3.grad(p)
ui = u + gp
ui_div = d3.div(ui)

solver = problem.build_solver()

scalars = solver.evaluator.add_file_handler(output_dir+'scalar', max_writes=np.inf, iter=100)
slices = solver.evaluator.add_file_handler(output_dir+'slices', max_writes=20, sim_dt=1.0)
slices.add_task(ui(phi=np.pi), name='ui_pi')
slices.add_task(ui(phi=0), name='ui_0')
scalars.add_task(ui_div, name='div_ui')
scalars.add_task(ui, name='ui')

solver.solve()
solver.evaluator.evaluate_handlers([scalars,slices])

print(np.max(np.abs(u['g'])))
print(np.max(np.abs(ui.evaluate()['g'])))
