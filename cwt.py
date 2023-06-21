from __future__ import division
from numpy import array,arange,zeros,linspace,loadtxt,convolve,mean,exp
from math import log,sin,sqrt,exp,pi,cos
import matplotlib.pyplot as plt
import cmath

#Parameter determining the "width" of the mother wavelet
#for the Morlet and Mexican hat wavelets
sigma = 8

#Morlet mother wavelet
def morlet(t):
    c = (1 + exp(-sigma**2) - 2*exp(-0.75*sigma**2))**-0.5
    k = exp(-0.5*sigma**2)
    wv = c * pi**-0.25 * exp(-0.5*t**2) * (cmath.exp(1j*sigma*t) - k)
    return wv

#Mexican hat mother wavelet
def mexhat(t):
    wv = (1 - (t/sigma)**2) * (exp(-0.5*(t/sigma)**2))
    wv *= 2/(sqrt(3*sigma) * pi**0.25)
    return wv

#Continuous wavelet transform
def cwt(signal, dt, startscale, endscale, scalesep, waveletfnc):
    
    #Initialize arrays
    N = len(signal)
    T = dt*N
    times = arange(0,T,dt)
    scaling_vals = arange(startscale,endscale,scalesep)
    coefficients = zeros((len(scaling_vals),N),dtype=complex)
    
    #CWT
    for i in range(len(scaling_vals)):
        s = scaling_vals[i]
        wavelet = array([waveletfnc((t-T/2)/s) for t in times])
        wavelet /= sqrt(s)
        signal -= mean(signal)
        coefficients[i] = convolve(wavelet,signal,'same')
        
    return abs(coefficients), scaling_vals
    
if __name__ == '__main__':
    #Test data
    testtimes = linspace(0,50,1000)
    dt = 50/1000
    testsignal = array([cos(1*x) for x in testtimes])
    testsignal[300:] = array([cos(2.4*x) for x in testtimes[300:]])
    testsignal[600:] += array([sin(5*x) for x in testtimes[600:]])
    
    #CWT of test data
    testimage, testscales = cwt(testsignal,dt,0.05,15,0.01,morlet)
    
    #fMRI data
    signal1 = loadtxt('subj001_DMN_PCC.txt')
    signal2 = loadtxt('subj001_ECN_LDLPFC.txt')
    dt1 = 2.5
    
    #CWT of fMRI data using Mexican hat wavelet
    #sigma=5
    #image2, scales2 = cwt(signal1,dt1,0.5,8,0.1,mexhat)
    #image3, scales3 = cwt(signal2,dt1,0.5,8,0.1,mexhat)
    
    #CWT of fMRI data using Morlet wavelet
    sigma=1
    image2, scales2 = cwt(signal1,dt1,3,35,0.1,morlet)
    image3, scales3 = cwt(signal2,dt1,3,35,0.1,morlet)
    
    #Plot results
    plt.figure(1)
    plt.subplot(211)
    plt.plot(testtimes,testsignal)
    plt.grid()
    plt.subplot(212)
    plt.imshow(testimage,aspect='auto',extent=[0,50,testscales[-1],testscales[0]])
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('CWT using Morlet wavelet')
    
    plt.figure(2)
    plt.subplot(211)
    plt.plot(arange(0,dt1*len(signal1),dt1),signal1)
    plt.xlim((0,dt1*len(signal1)))
    plt.title('Default mode network')
    plt.grid()
    plt.subplot(212)
    plt.imshow(image2,aspect='auto',extent=[0,len(signal1)*dt1,scales2[-1],scales2[0]])
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('CWT using Morlet wavelet')
    
    plt.figure(3)
    plt.subplot(211)
    plt.plot(arange(0,dt1*len(signal2),dt1),signal2)
    plt.xlim((0,dt1*len(signal2)))
    plt.title('Executive control network')
    plt.grid()
    plt.subplot(212)
    plt.imshow(image3,aspect='auto',extent=[0,len(signal2)*dt1,scales3[-1],scales3[0]])
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('CWT using Morlet wavelet')