# This script was made to make an easy way to plot the CIR from testing. The
# testing was to get all 1016 taps of the CIR response and then come up with
# an algorithm in python to find the first path index. Over many iterations,
# we found an algorithm that worked in many testing scenarios. The three 
# measurement campaigns are saved in the folder data/data_for_algorithm. In all
# campaigns, I moved the TX around while taking measurements. Some were through
# wall and some were line of sight.
#
# The algorithm we came up with to find the first path was to sort the CIR, and
# find the smallest index of the top 100 whose magnitude was greater than 2500.
# This proved to be a very robust way of finding the first path. This algorithm
# was then implemented in C and flashed on the EVB1000 RX.
import numpy as np
import matplotlib.pyplot as plt

# First find the maximum magnitude so that we can keep the y-axis the same for
# all plots
max_mag = 0
fname = 'data/data_for_algorithm/long_in_and_out.txt'
# Find max magnitude
with open(fname,'r') as f:
    for line in f:
        line = line.split(' ')
        complex_vals = np.array([float(ii) for ii in line[:-1]])
        #mag = complex_vals[0::2]**2 + complex_vals[1::2]**2
        mag = np.abs(complex_vals[0::2]) + np.abs(complex_vals[1::2])
        
        max_mag = np.maximum(max_mag,mag.max())

# Setup figures
fig, ax = plt.subplots()
is_first_ts = 1

max_in_noise = 0

# Run through each line from file
with open(fname,'r') as f:
    for line in f:
        line = line.split(' ')
        time_stamp = float(line[-1].split('\n')[0])
        if is_first_ts:
            start_time = time_stamp
            is_first_ts = 0
        
        complex_vals = np.array([float(ii) for ii in line[:-1]])
        #mags = complex_vals[0::2]**2 + complex_vals[1::2]**2
        mags = np.abs(complex_vals[0::2]) + np.abs(complex_vals[1::2])
        
        max_in_noise = np.maximum(mags[:650].max(),max_in_noise)
         
        sorted_idx = np.argsort(mags)
         
        # Test algorithm
        num_min = 100
        min_idx = 2000
        for ii in range(num_min):
            if (sorted_idx[-ii]<min_idx) & (mags[sorted_idx[-ii]]>2500):
                min_idx = sorted_idx[-ii]
          
        idx_to_show = np.arange(650,int(7*mags.size/8.))
    
        # plot CIR and the first path
        ax.clear()
        ax.plot(idx_to_show,mags[idx_to_show])
        ax.plot(min_idx-2,mags[min_idx-2],'ro',ms=10)
        ax.set_ylim(0,max_mag+1e3)
        ax.set_title('Frame ' + str(time_stamp-start_time))
        plt.show(block=False)
        plt.pause(0.01)
print max_in_noise