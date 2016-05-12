# This script runs through test 2 through 5 in data. It plots the distribution
# of the magnitude of one tap of the CIR over the duration of the experiment.
# The figure will show the on, off, and near link line distributions
import numpy as np
import cir_mag_class_master as cir
import matplotlib.pyplot as plt

# Off
fname = '2016_05_06_test_2.txt'
my_cir_obj = cir.cir_power_class(N_up=8,adj_type='pl')
pdp_mat = []

with open('data/cirs/' + fname,'r') as f:
    for line in f:
        
        if my_cir_obj.observe(line) == 0:
            continue
        
        pdp_mat.append(my_cir_obj.get_cir_mag().tolist())

pdp_mat = np.array(pdp_mat)
pdp_mat[pdp_mat == 0] = np.nan
pdp_mat_off = 20*np.log10(pdp_mat.T)

# On
fname = '2016_05_06_test_3.txt'
my_cir_obj = cir.cir_power_class(N_up=8,adj_type='pl')
pdp_mat = []

with open('data/cirs/' + fname,'r') as f:
    for line in f:
        
        if my_cir_obj.observe(line) == 0:
            continue
        
        pdp_mat.append(my_cir_obj.get_cir_mag().tolist())

pdp_mat = np.array(pdp_mat)
pdp_mat[pdp_mat == 0] = np.nan
pdp_mat_on = 20*np.log10(pdp_mat.T)

# Near 1
fname = '2016_05_06_test_4.txt'
my_cir_obj = cir.cir_power_class(N_up=8,adj_type='pl')
pdp_mat = []

with open('data/cirs/' + fname,'r') as f:
    for line in f:
        
        if my_cir_obj.observe(line) == 0:
            continue
        
        pdp_mat.append(my_cir_obj.get_cir_mag().tolist())

pdp_mat = np.array(pdp_mat)
pdp_mat[pdp_mat == 0] = np.nan
pdp_mat_near1 = 20*np.log10(pdp_mat.T)

# Near 2
fname = '2016_05_06_test_5.txt'
my_cir_obj = cir.cir_power_class(N_up=8,adj_type='pl')
pdp_mat = []

with open('data/cirs/' + fname,'r') as f:
    for line in f:
        
        if my_cir_obj.observe(line) == 0:
            continue
        
        pdp_mat.append(my_cir_obj.get_cir_mag().tolist())

pdp_mat = np.array(pdp_mat)
pdp_mat[pdp_mat == 0] = np.nan
pdp_mat_near2 = 20*np.log10(pdp_mat.T)

#
fig, axarr = plt.subplots(2, sharex=True)

for plt_idx in range(pdp_mat_off.shape[0]):
    axarr[0].clear()
    axarr[1].clear()
    da_bins = np.arange(40,110)
    axarr[0].set_title('CIR tap ' + str(plt_idx/8.) + ' ns')
    axarr[0].hist(pdp_mat_off[plt_idx,:],bins=da_bins,normed=True,label='Off',color='red')
    axarr[0].hist(pdp_mat_on[plt_idx,:],bins=da_bins,normed=True,label='On',color='blue',alpha=0.5)
    axarr[0].set_ylabel('Rel. Freq.')
    axarr[0].legend()
    axarr[0].set_ylim(0,0.6)
    axarr[1].hist(pdp_mat_near1[plt_idx,:],bins=da_bins,normed=True,label='Near1',color='green',alpha=0.5)
    axarr[1].hist(pdp_mat_near2[plt_idx,:],bins=da_bins,normed=True,label='Near2',color='yellow',alpha=0.5)
    axarr[1].set_xlabel('CIR tap value (dB)')
    axarr[1].set_ylabel('Rel. Freq.')
    axarr[1].legend()
    axarr[1].set_ylim(0,0.6)
    plt.show(block=False)
    plt.savefig('figures/hist_'+str(plt_idx/8.) + 'ns.pdf',bbox_inches='tight')
    plt.pause(0.5)





