# This class processes incoming CIR measurements. This class is responsible for
# parsing a new line, saving the time, first path index, and the CIR. The user
# can choose what kind of adjust is made to correct the signal
import numpy as np
from scipy import signal
import sys
import matplotlib.pyplot as plt    

class cir_class:
    
    # Initialization
    def __init__(self,N_up=1,adj_type='pl',sig_type='c'):
        if N_up < 0:
            sys.stderr.write('Upsample number must be greater than 0')
            quit()
        
        self.N_up = N_up # upsample number
        self.adj_type = adj_type # User defined adjustment to signal: l-lag, p-phase, pl-phase and lag
        self.sig_type = sig_type # user defined type of signal: 'c' operates on complex-valued signal, 'm' operates on magnitude
        
        self.cur_time = None # the time at which the CIR was taken
        self.start_time = None # The first time stamp
        self.cur_signal_raw = None # the complex-valued CIR
        self.cur_signal_up = None # The upsampled complex-valued CIR
        self.first_path_tap_idx = None # the index of the first path
        
        self.ref_cir = None # This is the reference CIR that we use for lag/phase adjusting
        self.avg_cir = None # This is the filtered CIR reference we use for lag/phase adjusting
        self.num_taps = None # Number of CIR taps in raw signal
        self.up_sig_len = None # Length of upsampled signal
        self.is_first_obs = 1 # Flag that indicates if this is the first CIR
        self.filter_on = 1 # Flag that indicates if we want to add measurements to a filter
        
        self.phases_vec = None # A vector of phase values to optimize over
        self.phases_mat = None # Tiled phase vector for quick optimization
        self.lag_vec = None # A vector of possible lags for lag adjustment
    
    # Run through all of the measurements and get the max magnitude
    def get_max_mag(self,fname):
        max_mag = 0
        with open(fname,'r') as f:
            for line in f:
                line = line.split(' ')
                complex_vals = np.array([float(ii) for ii in line[:-2]])
                mag = complex_vals[0::2]**2 + complex_vals[1::2]**2
                
                max_mag = np.maximum(max_mag,mag.max())
        return max_mag   
    
    # This method takes in the next observation and processes it. It returns a
    # 1 if the measurement is good. Otherwise it returns a 0
    def observe(self,cur_obs):
        
        # parse observation
        self.parse_line(cur_obs)
            
        # check if it is a good CIR
        if self.is_good_cir() == 0:
            return 0
        
        # upsample the signal
        self.upsample_sig()
        
        # Adjust signal
        self.adjust_signal()
        
        # Add signal to filter
        self.filter_sig()
        
        # Return a 1 to report a successful processing
        return 1
    
    # Return the relative time
    def get_rel_time(self):
        return self.cur_time-self.start_time
    
    # Return the upsampled CIR magnitude
    def get_cir_mag(self):
        return self.sig_mag(self.cur_signal_up)
    
    # Return the raw CIR
    def get_raw_cir(self):
        return self.cur_signal_raw.copy()
    
    # Return the index of the first path
    def get_first_path_idx(self):
        return self.first_path_tap_idx
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Helper functions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #       
        
    # Parse a new line. Save sample time, first path index, and complex-valued
    # CIR
    def parse_line(self,cur_obs):
        # take off new line character
        cur_obs = cur_obs.split('\n')[0]
        
        # Split up the string by spaces
        cur_obs_split = cur_obs.split(' ')
        
        # Get the time stamp
        self.cur_time = float(cur_obs_split.pop())
        
        # Get the first path tap index
        self.first_path_tap_idx = int(float(cur_obs_split.pop()))
        
        # Convert str to floats for real and imag
        cur_R_raw = np.array([float(x) for x in cur_obs_split[0::2]])
        cur_I_raw = np.array([float(x) for x in cur_obs_split[1::2]])
        self.cur_signal_raw = cur_R_raw + 1j*cur_I_raw            
    
    # This method returns a 1 if the CIR measurement is "good". A good CIR has
    # at least one absolute magnitude value greater than 3000. Otherwise, we
    # know that the EVB1000 found the first path index in the middle of the 
    # noise floor.
    def is_good_cir(self):

        # Add the absolute complex values together
        abs_sig = np.abs(np.real(self.cur_signal_raw)) + np.abs(np.imag(self.cur_signal_raw))
        
        # Through empirical observation, we found that it was very rare
        # to see an |real| + |imag| value greater than 2500 in the noise
        # floor.
        if abs_sig.max() > 3000:
            # This CIR has at least one non-noise floor sample
            return 1
        else:
            # This CIR has all noise floor samples
            return 0

    # Adjust signal according to users specifications
    def adjust_signal(self):
        if self.adj_type == 'l':
            self.adjust_for_lag()
        elif self.adj_type == 'p':
            self.adjust_for_phase()
        elif self.adj_type == 'pl':
            self.adjust_for_phase_and_lag()

    # This method upsamples the complex-valued CIR
    def upsample_sig(self):
        
        # Check that the upsample value is more than 0
        # upsample the signal
        self.cur_signal_up = signal.resample(self.cur_signal_raw, self.cur_signal_raw.size*self.N_up)
        tmpr = signal.resample(np.real(self.cur_signal_raw), self.cur_signal_raw.size*self.N_up)
        
        # change signal to magnitude of signal according to user specs
        if self.sig_type == 'm':
            self.cur_signal_up = self.sig_mag(self.cur_signal_up)
        
        # if the user asked to filter, add observation to average
        if self.filter_on:
            self.filter_sig()
        
        # if this is our first CIR, save it for reference, and flip flag
        if self.is_first_obs:
            self.start_time = self.cur_time
            self.num_taps = self.cur_signal_raw.size
            self.up_sig_len = self.cur_signal_up.size
            self.ref_cir = self.cur_signal_up.copy()
            
            self.__init_mats()
            
            self.is_first_obs = 0
        
    # This method filters the signal with an exponentially decaying moving average
    def filter_sig(self):
        alpha = 0.05
        
        # If this is the first observation
        if self.is_first_obs:
            self.avg_cir = self.cur_signal_up
        else:
            self.avg_cir = alpha*self.cur_signal_up + (1-alpha)*self.avg_cir 
        
            
    # This method adjusts for any lag in the upsampled signal
    def adjust_for_lag(self):
        y = self.cur_signal_up.copy()
         
        x = self.sig_mag(y)
         
        corr_mag = np.correlate(x,self.sig_mag(self.avg_cir),mode='full')
        max_mag_idx = np.argmax(corr_mag).flatten()[0]
         
        corr_cpx = np.correlate(y,self.avg_cir,mode='full')
        max_cpx_idx = np.argmax(self.sig_mag(corr_cpx)).flatten()[0]
         
        lags = np.arange(self.up_sig_len*2-1)-(self.up_sig_len-1)
        plt.plot(lags,corr_mag,'k')
        plt.plot(lags[max_mag_idx],corr_mag[max_mag_idx],'ko')
        plt.plot(lags,self.sig_mag(corr_cpx),'r')
        plt.plot(lags[max_cpx_idx],self.sig_mag(corr_cpx)[max_cpx_idx],'ro')        
        plt.show()
        
        
                    
        # copy the current CIR
        y = self.cur_signal_up.copy()
        
        # Auto-correlate signal according to user settings 
        if self.filter_on:
            cur_corr = np.correlate(y,self.avg_cir,mode='full')
        else:
            cur_corr = np.correlate(y,self.ref_cir,mode='full')
        opt_lag = -self.lag_vec[np.argmax(self.sig_mag(cur_corr)).flatten()[0]]
        
        # Shift the signal to adjust for any lag
        if opt_lag > 0:
            self.cur_signal_up = np.array(((0+1j*0)*np.ones(opt_lag)).tolist() + y[0:-opt_lag].tolist())
        elif opt_lag < 0:
            self.cur_signal_up = np.array(y[-opt_lag:].tolist() + ((0+1j*0)*np.ones(-opt_lag)).tolist())
        else:
            self.cur_signal_up = 1.0*y
        
        # Adjust for magnitude setting
        if self.sig_type == 'm':
            self.cur_signal_up = self.sig_mag(self.cur_signal_up)
    
    # This method adjust for any phase difference in the upsampled signal
    def adjust_for_phase(self):
        
        # Tile the upsampled signal
        sig_tiled = np.tile(self.cur_signal_up,(self.phases_vec.size,1))
        
        # Tile the reference signal
        if self.filter_on:
            ref_tiled = np.tile(self.avg_cir,(self.phases_vec.size,1))
        else:
            ref_tiled = np.tile(self.ref_cir,(self.phases_vec.size,1))
        
        # Multiply in phase in subtract out reference
        A = sig_tiled*self.phases_mat
        
        if self.sig_type == 'm':
            A = self.sig_mag(A)
        
        C = A - ref_tiled
        
        # Compute l2-norm of each row
        cur_norms = (self.sig_mag(C)).sum(axis=1)
        
        # Get the index with the least l2-norm
        opt_phase_idx = np.argmin(cur_norms).flatten()[0]
        
        # Adjust upsampled signal
        self.cur_signal_up = A[opt_phase_idx,:]
    
    # This method adjusts for phase and lag in the upsampled signal
    def adjust_for_phase_and_lag(self):
        # store a list of the l2-norms and the lags for each phase value
        norm_list = np.zeros((self.phases_vec.size,2))
        
        # loop through each phase value
        for pp in range(self.phases_vec.size):
            y = self.cur_signal_up.copy()
            y = y*np.exp(1j*self.phases_vec[pp])
            
            # Adjust for magnitude setting
            if self.sig_type == 'm':
                y = self.sig_mag(y)
            
            # compute autocorrelation
            if self.filter_on:
                cur_corr = np.correlate(y,self.avg_cir,mode='full')
            else:
                cur_corr = np.correlate(y,self.ref_cir,mode='full')
            opt_lag = -self.lag_vec[np.argmax(self.sig_mag(cur_corr)).flatten()[0]]
            norm_list[pp,0] = opt_lag
            
            # Shift the signal to adjust for any lag
            if opt_lag > 0:
                y = np.array(((0.+1j*0.)*np.ones(opt_lag)).tolist() + y[0:-opt_lag].tolist())
            elif opt_lag < 0:
                y = np.array(y[-opt_lag:].tolist() + ((0.+1j*0.)*np.ones(-opt_lag)).tolist())
            
            # Adjust for magnitude setting
            if self.sig_type == 'm':
                y = self.sig_mag(y)
            
            # Compute the l2-norm
            if self.filter_on:
                tmp = y - self.avg_cir
            else:
                tmp = y - self.ref_cir
            
            # Save l2-norm to list
            norm_list[pp,1] = self.sig_mag(tmp).sum()
        
        # Get the index of the smallest l2-norm
        min_idx = np.argmin(norm_list[:,1]).flatten()[0]
        
        # Adjust for phase and lag
        y = self.cur_signal_up.copy()
        y = y*np.exp(1j*self.phases_vec[min_idx])
        opt_lag = norm_list[min_idx,0]
        
        # Shift the signal to adjust for any lag
        if opt_lag > 0:
            self.cur_signal_up = np.array(((0+1j*0)*np.ones(opt_lag)).tolist() + y[0:-opt_lag].tolist())
        elif opt_lag < 0:
            self.cur_signal_up = np.array(y[-opt_lag:].tolist() + ((0+1j*0)*np.ones(-opt_lag)).tolist())
        else:
            self.cur_signal_up = y.copy()
            
        # Adjust for magnitude setting
        if self.sig_type == 'm':
            self.cur_signal_up = self.sig_mag(self.cur_signal_up)
        
    # Initialize frequently used vectors and matrixes
    def __init_mats(self):
        # Get a phases vector and make a phasor matrix for phase adjustment
        self.phases_vec = np.linspace(0,2*np.pi,endpoint=False,num=24)
        self.phases_mat = np.tile(np.exp(1j*self.phases_vec),(self.up_sig_len,1)).T
        
        # Get a lag vector
        self.lag_vec = np.arange(self.up_sig_len*2-1)-(self.up_sig_len-1)
        
    # compute magnitude of signal
    def sig_mag(self,sig):
        return np.sqrt(np.real(sig)**2 + np.imag(sig)**2)

# # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # 

# This class allows the user to quickly access the power in the first path
class cir_power_class(cir_class):
        
    # Get power in the first path (defined as the power in the first three ns.
    # Each tap is 1 ns
    def get_first_path_power(self):
        idx_3ns = np.arange(self.first_path_tap_idx,self.first_path_tap_idx+3+1)*self.N_up
        y= self.sig_mag(self.cur_signal_up[idx_3ns])
        return 20*np.log10(y.sum())
    
    # Get power in all taps after first path
    def get_full_power(self):
        y = self.sig_mag(self.cur_signal_up[self.first_path_tap_idx*self.N_up:])
        return 20*np.log10(y.sum())
        
        
    
    
    