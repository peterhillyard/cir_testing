# This script plots the CIR pdp and the power over time. This script is intended
# for plotting the results of the 2016_05_06 experiments where I did the 
# on, off, and nearby the link line experiments. These experiments only consider
# a link in one location. Only the environment changed.
import numpy as np
# import cir_mag_class_master as cir
import matplotlib.pyplot as plt
import os
import sys 

sys.path.insert(0, '../my_span_package') # Adds higher directory to python modules path.
import cir_class as cir

# Parameters
fname = '2016_05_06_test_1.txt'
# my_cir_obj = cir.cir_power_class(N_up=8,adj_type='l',sig_type='c')
my_cir_obj = cir.cir_power_class(N_up=8,adj_type='pl')
power_vec = []
pdp_mat = []
phase_mat = []
time_vec = []
 
fig, ax = plt.subplots()
 
with open('data/cirs/' + fname,'r') as f:
    for line in f:
         
        if my_cir_obj.observe(line) == 0:
            continue
         
#         y = my_cir_obj.cur_signal_up.copy()
#         ax.plot(np.real(y),np.imag(y))
#         plt.show(block=False)
#         plt.pause(0.5)
         
        power_vec.append(my_cir_obj.get_full_power())
        pdp_mat.append(my_cir_obj.get_cir_mag().tolist())
        phase_mat.append(my_cir_obj.get_phase().tolist())
        time_vec.append(my_cir_obj.get_rel_time())
 
power_vec = np.array(power_vec)
pdp_mat = np.array(pdp_mat).T
phase_mat = np.array(phase_mat).T
time_vec = np.array(time_vec)
 
y_max = my_cir_obj.num_taps
xing_y_val = my_cir_obj.first_path_tap_idx
 
if os.path.isfile('data/xings/' + fname):
    xings = np.loadtxt('data/xings/' + fname)
else:
    xings = None
 
plt_idx = 6*8+7
plt.subplot(3,1,1)
plt.plot(time_vec,(phase_mat[plt_idx,:]),'k-')
plt.ylabel('Phase (radian)')
plt.xlabel('Time (sec)')
plt.grid()
plt.subplot(3,1,2)
if xings is not None:
    plt.plot(xings,np.zeros(xings.size),'rx',ms=10,mew=10)
plt.plot(time_vec,np.unwrap(phase_mat[plt_idx,:]),'k-')
plt.ylabel('Phase (radian)')
plt.xlabel('Time (sec)')
plt.grid()
plt.subplot(3,1,3)
if xings is not None:
    plt.plot(xings,np.zeros(xings.size),'rx',ms=10,mew=10)
plt.plot(time_vec[:-1],np.diff(np.unwrap(phase_mat[plt_idx,:])),'k-')
plt.ylabel('Difference in Phase (radian)')
plt.xlabel('Time (sec)')
plt.grid()
plt.show(block=False)
 
# plt.figure()
# tmp = 20*np.log10(pdp_mat[:,plt_idx])
# plt.plot(tmp-tmp.max())
# plt.xlabel('Excess Delay')
# plt.ylabel('dB')
# plt.grid()
# plt.show(block=True)
 
# Plot mag and phase in the same figure
plt.figure()
plt.subplot(2,1,1)
plt.imshow(20*np.log10(pdp_mat),origin='upper',aspect='auto',interpolation='none',
           extent = (time_vec[0],time_vec[-1],y_max,0))
cbar = plt.colorbar()
cbar.set_label('dB')
if xings is not None:
    plt.plot(xings,xing_y_val*np.ones(xings.size),'rx',ms=10,mew=10)
plt.xlim(time_vec[0],time_vec[-1])
plt.xlabel('Time (sec)')
plt.ylabel('Time delay (ns)')
plt.subplot(2,1,2)
plt.imshow(np.unwrap(phase_mat,axis=1),origin='upper',aspect='auto',interpolation='none',
           extent = (time_vec[0],time_vec[-1],y_max,0),clim=(-4, 4))
plt.colorbar()
cbar.set_label('Phase (rad)')
if xings is not None:
    plt.plot(xings,xing_y_val*np.ones(xings.size),'rx',ms=10,mew=10)
plt.xlim(time_vec[0],time_vec[-1])
plt.xlabel('Time (sec)')
plt.ylabel('Time delay (ns)')
plt.show()
 
# Plot power
# plt.figure()
# plt.plot(time_vec,power_vec)
# if xings is not None:
#     plt.plot(xings,np.median(power_vec)*np.ones(xings.size),'rx',ms=10,mew=10)
# plt.xlim(time_vec[0],time_vec[-1])
# plt.xlabel('Time (sec)')
# plt.ylabel('Power (dB)')
# plt.show()









