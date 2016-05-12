# This script runs on a BBB to collect the real and imaginary CIR measurements
# from the EVB1000 RX. The user can change the number of taps read in if the 
# code flashed on the RX changes. The user provides the file name in the command
# line so the measurements can be saved to file.

import Adafruit_BBIO.UART as UART
import serial
import time
import sys

class my_cir():

    # Constructor
    def __init__(self,num_cir_taps,baud_rate,fname):
        self.ser = None # serial object to pull in UART data
        self.currentLine = [] # holds the data from UART
        self.first_path_tap_idx = None
        
        self.baud_rate = baud_rate # baud rate of UART
        self.num_cir_taps = num_cir_taps # number of taps of CIR sent from UART
        # Length of a packet - number of taps times two complex values, times 
        # two bytes per complex value, plus the first path tap index, plus 4 
        # terminating values
        self.packet_size = self.num_cir_taps*2*2+1+4
        self.fname = fname # Name of the file - do not include .txt
        self.fout = None # output file

        self.prev_time = time.time()

        self.__initialize_uart()
        self.__open_file()

    # Observe function
    def observe(self):
        cur_obs = self.ser.read().encode('hex')
        #print cur_obs
        self.currentLine.append(cur_obs)

        # Whenever the end-of_line sequence is read, operate on the "packet" of data
        if (self.currentLine[-4:] == ['44','45', '41', '44']): # "The corresponding string is 'DEAD' "
            # the packet size is correct
            if len(self.currentLine) == self.packet_size:
                #print self.currentLine
                
                # Order from high bits to low bits
                self.__remove_and_reorder()
                
                # Convert 2's complement hex to integer
                self.__hex_to_int()
                
                # print line to file
                self.__print_to_file()
                
            self.currentLine = []

    ##################################################
    # Helper Functions                                     
    
    # remove terminating values and reorder the list
    def __remove_and_reorder(self):
        # Remove terminating values
        [self.currentLine.pop() for x in range(4)]
        
        # Get the first path tap index
        self.first_path_tap_idx = self.currentLine.pop()
        self.first_path_tap_idx = float(int(self.first_path_tap_idx,16))
        # print self.first_path_tap_idx
        
        # Get the CIR complex values
        low_bits  = self.currentLine[0::2]
        high_bits = self.currentLine[1::2]
        #print self.currentLine
        self.currentLine = ['0x' + high_bits[x] + low_bits[x] for x in range(self.num_cir_taps*2)]
        #print self.currentLine
    
    # compute 2's complement hex to integer
    def __hex_to_int(self):
        self.currentLine = [float(self.__twos_comp(int(item,16), 16)) for item in self.currentLine]
        #print self.currentLine
    
    # print real and imag integers to file
    def __print_to_file(self):
        for item in self.currentLine:
            self.fout.write("%s " % item)
        self.fout.write(str(self.first_path_tap_idx) + ' ')
        self.fout.write(str(time.time()) + '\n')
        print time.time()-self.prev_time
        self.prev_time=time.time()
    
    # Convert hex string to integer
    def __twos_comp(self, val, bits):
        """compute the 2's compliment of int value val"""
        if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)        # compute negative value
        return val

    # Initialize UART and serial port
    def __initialize_uart(self):
        UART.setup("UART1")
        self.ser = serial.Serial(port = "/dev/ttyO1", baudrate=self.baud_rate)
        self.ser.close()
        self.ser.open()

    # Open file object        
    def __open_file(self):
        self.fout = open('data/' + self.fname + '.txt','w')
        


###########################
# Start main program
num_cir_taps = 20
baud_rate = 57600
my_cir_obj = my_cir(num_cir_taps,baud_rate,sys.argv[1])
print "starting script"

while(1):
    my_cir_obj.observe()






