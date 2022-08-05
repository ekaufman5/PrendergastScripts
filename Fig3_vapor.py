"""
Reads in 3-D ball data and interpolates to a uniform cartesian grid for 3-D vis.

Usage:
    uniform_cartesian_flows_netcdf.py <file> [options]

Options:
    --field=<field>         Name of field to interpolate for visualization [default: u]
    --n=<n>                 n^3 resolution of uniform cartesian cube [default: 256]
"""
from docopt import docopt
args = docopt(__doc__)

filename = args['<file>']
field = args['--field']
n = int(float(args['--n']))

nx = ny = nz = n

import logging

import numpy as np
import h5py
import time
import cfdm

logger = logging.getLogger(__name__)

f = h5py.File('{:s}'.format(filename), 'r')

task = f['tasks'][field]
t = task.dims[0][0][:]
phi = task.dims[1][0][:]
theta = task.dims[2][0][:]
r = task.dims[3][0][:]

r = r[None,None,:]
theta = theta[None,:,None]
# wrap phi points to ensure good interpolation properties in periodic direction
phi = np.concatenate((phi, [phi[0]+2*np.pi]))
phi = phi[:,None,None]

print("opening {}".format(filename))
print(" r:{}".format(r.shape))
print("th:{}".format(theta.shape))
print("ph:{}".format(phi.shape))

x = r*np.sin(theta)*np.cos(phi)
y = r*np.sin(theta)*np.sin(phi)
z = r*np.cos(theta)*np.ones_like(phi)

x_u = np.linspace(-1, 1, nx)[:,None,None]
y_u = np.linspace(-1, 1, ny)[None,:,None]
z_u = np.linspace(-1, 1, nz)[None,None,:]

zero = np.zeros((nx,ny,nz))
r_u = zero + (x_u**2 + y_u**2 + z_u**2)**0.5
theta_u = zero + np.arccos(z_u/r_u)
phi_u = zero + np.arctan2(y_u,x_u)
phi_u[phi_u<0] += 2*np.pi

from scipy.interpolate import RegularGridInterpolator
original_coords_flat = (phi.flatten(), theta.flatten()[::-1], r.flatten())
points = np.array([phi_u.flatten(), theta_u.flatten(), r_u.flatten()]).T

domain_axisT = cfdm.DomainAxis(1)
domain_axisx = cfdm.DomainAxis(nx)
domain_axisy = cfdm.DomainAxis(ny)
domain_axisz = cfdm.DomainAxis(nz)

dimT = cfdm.DimensionCoordinate(properties={'standard_name': 'time', 'axis':'T','units': 's'}, data=cfdm.Data(t))
dimx = cfdm.DimensionCoordinate(properties={'standard_name': 'x','axis':'X', 'units': 'm'}, data=cfdm.Data(np.squeeze(x_u)))
dimy = cfdm.DimensionCoordinate(properties={'standard_name': 'y','axis':'Y', 'units': 'm'}, data = cfdm.Data(np.squeeze(y_u)))
dimz = cfdm.DimensionCoordinate(properties={'standard_name': 'z','axis':'Z', 'units': 'm'}, data=cfdm.Data(np.squeeze(z_u)))

task_data = {}
for i, component in enumerate(['ph', 'th', 'r']):
    task_data[field+'_'+component] = task[0,i,:,:,:]
task_data[field+'_mag'] = np.sqrt(np.sum(task[0,:,:,:,:]**2, axis=0))

for task in task_data:
    print(task_data[task].shape)
    task_data[task] = np.append(task_data[task][:,:,:], np.expand_dims(task_data[task][0,:,:], axis=0), axis=0)

task_data[field+'_x'] = task_data[field+'_r']*np.sin(theta)*np.cos(phi) + task_data[field+'_th']*np.cos(theta)*np.cos(phi) - task_data[field+'_ph']*np.sin(phi)
task_data[field+'_y'] = task_data[field+'_r']*np.sin(theta)*np.sin(phi) + task_data[field+'_th']*np.cos(theta)*np.sin(phi) + task_data[field+'_ph']*np.cos(phi)
task_data[field+'_z'] = task_data[field+'_r']*np.cos(theta) - task_data[field+'_th']*np.sin(theta)

data_set = []
for label in task_data:
    print(label)
    data = cfdm.Field( properties={'standard_name': label,'units': '1'})
    axisx = data.set_construct(domain_axisx)
    axisy = data.set_construct(domain_axisy)
    axisz = data.set_construct(domain_axisz)

    data.set_construct(dimx, axes=axisx)
    data.set_construct(dimy, axes=axisy)
    data.set_construct(dimz, axes=axisz)

    F_interp = RegularGridInterpolator(original_coords_flat,np.array(task_data[label])[:,::-1,:],
                                       bounds_error=False, fill_value=0)

    start_int = time.time()
    data_u = F_interp(points).reshape((nx, ny, nz))
    end_int = time.time()
    print("Interpolation took {:g} seconds for {}".format(end_int-start_int, label))

    # Switch x and z directions... fortran memory ordering?
    data.set_data(data_u.T, axes=[axisz, axisy, axisx])

    data_set.append(data)

cfdm.write(data_set, 'vapor_B_data.nc')

