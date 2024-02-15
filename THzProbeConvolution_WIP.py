from scipy.signal import gausspulse as pulse, convolve
import numpy as np
from matplotlib import pyplot as plt
import random
import time

testprobe = [0,1,2,3,2,1,0]
testthz = [0,0,0,0,0,0,0,1,2,3,4,5,6,8,10,12,16,18,19,19.5,20,19.5,19,18,16,12,10,8,6,5,4,3,2,1,0,0,-1,-2,-3,-4,-6,-7,-8,-7,-6,-4,-3,-2,-1,0,0,1,2,3,2,1,0,0,0,0,0,0,0,0,0,1,2,3,2,1,0,0,0,0,0,0,0,0]

def generatePulses(f_thz=5, f_probe=500, f_sampling=250):

    t_probe = np.linspace(-0.1,0.1,f_sampling)
    p,probe = pulse(t_probe,f_probe,retenv=True)

    t_thz = np.linspace(-1,1,f_sampling)
    t,thz = pulse(t_thz,f_thz,retenv=True)

    thz_trace = np.zeros(f_sampling)
    for i, point in enumerate(thz):
        thz_trace[i] = point

    probePulse = np.ndarray.tolist(probe)
    probePulse = [p for p in probe if p > 0.00001]
    thzPulse = np.ndarray.tolist(thz_trace)

    return(probePulse,thzPulse)

def padTHz(thz, pad):
    thzLen = len(thz)
    print(f"THz trace length before padding: {thzLen}")
    for i in range(pad):
        thz.insert(0,0)
        thz.append(0)
    print(f"THz trace length after padding: {len(thz)}")

def noisyConvolution(probe,thz,samples,pad,amplNoise,timeNoise=0):
    print(f"Padding = {pad}")
    print(f"Probe length = {len(probe)}")
    startIndex = int(pad - np.floor(len(probe)/2))
    # startIndex = 0 #not sure why this is required...
    thzLen = len(thz)
    # print(f"THz trace length before padding: {thzLen}")
    # for i in range(pad):
    #     thz.insert(0,0)
    #     thz.append(0)
    # print(f"THz trace length after padding: {len(thz)}")
    convolved = []
    std = []
    random.seed(time.time())
    for i in range(thzLen-pad*2):
        currentValue = []
        for n in range(samples):
            dA = random.uniform(1-amplNoise, 1+amplNoise) #without amplitude noise, multiplier should be 1, so dA is centered around 1
            dt = int(random.uniform(0-timeNoise, 0+timeNoise)) #without time jitter, dt should be 0, so dt centered around 0
            # dt = timeNoise
            noisyProbe = [p*dA for p in probe]
            currentIndex = startIndex + i + dt
            currentValue.append(np.dot(noisyProbe, thz[currentIndex:currentIndex+len(noisyProbe)]))
        # print(f"Position {i} completed.")
        convolved.append(np.average(currentValue))
        std.append(np.std(currentValue))
    print('noisyConvolution() finished')
    return(convolved, std)

padding = 100
samples = 100
noise = 0.5
timeJitter = 5

probeData, thzData = generatePulses()
padTHz(thzData, padding)

print('beginning noisyConvolution()')
# noisyConvol, noisyStd = noisyConvolution(probeData,thzData,samples,int(np.floor(len(probeData)/2)),noise)
controlConvol, controlStd = noisyConvolution(probeData,thzData,samples,padding,0,0)

convol1, std1 = noisyConvolution(probeData, thzData, samples, padding, 0.2, 0)
convol2, std2 = noisyConvolution(probeData, thzData, samples, padding, 0.5, 0)
convol3, std3 = noisyConvolution(probeData, thzData, samples, padding, 0.7, 0)
convol4, std4 = noisyConvolution(probeData, thzData, samples, padding, 0.1, 0)
convol5, std5 = noisyConvolution(probeData, thzData, samples, padding, 1.5, 0)

convol6, std6 = noisyConvolution(probeData, thzData, samples, padding, 0, 2)
convol7, std7 = noisyConvolution(probeData, thzData, samples, padding, 0, 3)
convol8, std8 = noisyConvolution(probeData, thzData, samples, padding, 0, 5)
convol9, std9 = noisyConvolution(probeData, thzData, samples, padding, 0, 7)
convol10, std10 = noisyConvolution(probeData, thzData, samples, padding, 0, 10)

# print('beginning standard convolution')
# compareFull = convolve(probeData,thzData,mode='full')
# compareValid = convolve(probeData,thzData,mode='valid')

# snr = [abs(s/n) for s,n in zip(noisyConvol,noisyStd)]

plt.figure(2)
plt.plot(controlConvol)
plt.plot(convol1)
plt.plot(convol2)
plt.plot(convol3)
plt.plot(convol4)
plt.plot(convol5)
plt.plot(convol6)
plt.plot(convol7)
plt.plot(convol8)
plt.plot(convol9)
plt.plot(convol10)
plt.legend(["No noise","dA = 0.2","dA = 0.5","dA = 0.7","dA = 1","dA = 1.5","dt = 2","dt = 3","dt = 5","dt = 7","dt = 10"])
# plt.plot(snr, 'c:')
plt.show()

print('done')