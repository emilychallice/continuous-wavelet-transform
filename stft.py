from __future__ import division
from numpy import zeros,array,arange,linspace,pi,sin,cos,mean,loadtxt
from numpy.fft import fft
import matplotlib.pyplot as plt

#Hann window function
def hann(t,width):
    t += width/2
    if 0<=t<=width:
        return 0.5 * (1-cos(2*pi*t/(width)))
    else:
        return 0

#Short-Time Fourier Transform
def stft(signal,dt,windowlength):
    N = len(signal)
    T = N*dt
    times = arange(0,T,dt)
    coefficients = zeros((N, N//2))

    for i in range(N):
        window = array([hann(t-i*dt, windowlength) for t in times])
        multiplied = signal*window
        ft = fft(multiplied)[:N//2]
        coefficients[i] = abs(ft)
        
    return coefficients

if __name__ == '__main__':
    #Test signal
    testtimes = linspace(0,10,1024)
    testsignal= array([sin(30*x) for x in testtimes])
    testsignal[512:] = array([sin(70*x) for x in testtimes[512:]])
    dt = 10/1024
    
    #fMRI data
    signal1 = loadtxt('subj001_DMN_PCC.txt')
    signal2 = loadtxt('subj001_ECN_LDLPFC.txt')
    signal1 -= mean(signal1)
    signal2 -= mean(signal2)
    
    #Sampling rate of the fMRI data is 2.5s
    dt1 = 2.5
    
    #Compute STFTs for the three signals
    image1 = stft(testsignal,dt,2)
    image2 = stft(signal1,dt1,30)
    image3 = stft(signal2,dt1,30)
    
    #Plot results
    plt.figure(1)
    plt.subplot(211)
    plt.plot(testtimes,testsignal)
    plt.xlim((0,dt*len(testsignal)))
    plt.subplot(212)
    plt.imshow(image1.T,aspect='auto',origin='lower',
               extent=[0,20,0,len(testsignal)//2])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('STFT, window width = 2 s')
    
    plt.figure(2)
    plt.subplot(211)
    plt.plot(arange(0,dt1*len(signal1),dt1),signal1)
    plt.xlim(0,dt1*len(signal1))
    plt.title('Default mode network')
    plt.grid()
    plt.subplot(212)
    plt.imshow(image2.T,aspect='auto',origin='lower',
               extent=[0,dt1*len(signal2),0,len(signal1)//2])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('STFT, window width = 30 s')
    
    plt.figure(3)
    plt.subplot(211)
    plt.plot(arange(0,dt1*len(signal2),dt1),signal2)
    plt.xlim(0,dt1*len(signal2))
    plt.title('Executive control network')
    plt.grid()
    plt.subplot(212)
    plt.imshow(image3.T,aspect='auto',origin='lower',
               extent=[0,dt1*len(signal1),0,len(signal2)//2])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('STFT, window width = 30 s')