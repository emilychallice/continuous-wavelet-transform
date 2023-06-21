from __future__ import division
from math import sin,cos,pi
from numpy.fft import fft,ifft
from time import clock
import matplotlib.pyplot as plt
import numpy as np

#"Manual" convolution of signals a and b
def convolve(a,b,animate=False):
    #Pre-compute lengths
    len_a = len(a)
    len_b = len(b)
    
    #Force the shorter array to be a
    if len(a) >= len(b):
        a,b = b,a
        len_a,len_b = len_b,len_a
        
    #Initialize array for convolution
    convolution = np.zeros(len_b+2*len_a)
    
    #Flip a
    a = a[::-1]
    
    #Zero-pad b
    b_padded = np.zeros(len_b+2*len_a)
    b_padded[len_a-1:len_a+len_b-1] = b-np.mean(b)
    
    #Shift a along b
    for i in range(len_b + len_a):
        a_shifted = np.zeros(len_b+2*len_a)
        a_shifted[i:i+len_a] = a
        
        if animate:
            #Animate b and the shifting a
            plt.subplot(211)
            plt.plot(b_padded)
            plt.plot(a_shifted)
            plt.grid()
            plt.xlim(0,len_b+2*len_a)
            ylimit = max(max(abs(a)),max(abs(b))) + 1
            plt.ylim((-ylimit,ylimit))
        
        #Compute convolution
        multiplied = a_shifted*b_padded
        convolution[i+int(len_a/2)] = sum(multiplied)
        
        if animate:
            #Animate multiplied signal
            line1, = plt.plot(multiplied,'g',linestyle='--')
            plt.legend([line1],['Multiplied \nsignals'],fontsize=12)
            
            #Animate convolution
            plt.subplot(212)
            line2, = plt.plot(convolution,'r')
            plt.legend([line2],['Convolution'],fontsize=12)
            plt.grid()
            plt.xlim(0,len_b+2*len_a)
            
            #Not precomputed, vary accordingly; this limit is appropriate
            #for the default test signals in this module.
            plt.ylim((-40,40))
            
            plt.pause(0.01)
            plt.clf()
        
    return convolution[len_a//2:len_b+3*len_a//2-1]

#Convolution of a and b using FFT method
def convolve_fft(a,b):
    len_a,len_b = len(a),len(b)
    a_padded = np.zeros(2*(len_a+len_b))
    b_padded = np.copy(a_padded)
    a_padded[len_b:len_b+len_a] = a-np.mean(a)
    b_padded[len_a:len_a+len_b] = b-np.mean(b)
    
    convolution = ifft(fft(a_padded)*fft(b_padded))
    return convolution[len_a+len_b:2*(len_a+len_b)-1]
    

#Testing differences between three methods
if __name__ == '__main__':
    do = raw_input('Enter "compare" to compare manual, fft, numpy methods.\n'
    'Enter "animate" for convolution visualization.\n'
    'Enter anything else to exit.\n> ')
    
    #Test signals
    signal1 = np.array([-sin(x) for x in np.arange(0,2*pi,0.1)])
    signal2 = np.array([cos(x) for x in np.arange(-3.5*pi,2.5*pi,0.1)])   
    
    if do=='compare':
        t1=clock()
        convolution_manual = convolve(signal1,signal2)
        t2=clock()
        print '\nManual convolution:', t2-t1,'s'
        
        t1=clock()
        convolution_fft = convolve_fft(signal1,signal2)
        t2=clock()
        print 'FFT convolution:',t2-t1,'s'
        
        t1=clock()
        convolution_numpy = np.convolve(signal1,signal2)
        t2=clock()
        print 'Numpy convolution:',t2-t1,'s'
        
        line1, = plt.plot(convolution_manual+3)
        line2, = plt.plot(convolution_fft.real-3)
        line3, = plt.plot(convolution_numpy)
        plt.legend([line1,line2,line3],['Manual\n(shifted up)','FFT\n(shifted down)',
                                        'Numpy'],fontsize=12)
        plt.title('Convolution')
        plt.grid()
        plt.show()
        
    elif do=='animate':
        convolve(signal1,signal2,True)