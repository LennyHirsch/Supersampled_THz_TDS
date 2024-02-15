from scipy.signal import gausspulse
import matplotlib.pyplot as plt
import numpy as np

#custom gaussian function for single pulse probe generation

def singleGaussPulse(array, amplitude = 1, center = 0, width = 0.5):
    y = []
    for x in array: y.append(amplitude*np.exp(-((x-center)**2)/(0.5*width**2)))
    return y

x = np.linspace(0,400e-15,1000)
y = singleGaussPulse(x)
y2 = singleGaussPulse(x, 100, 200e-15, 100e-15)


plt.figure(1)
plt.grid(True, 'both')
plt.plot(x,y, 'r')
plt.plot(x,y2, 'b')

pulseTime = 1.25e-12
# thzDuration = 500e-16
# thzFreq = 1/thzDuration
thzFreq = 2.5e12
thzDuration = 1/thzFreq
print(thzDuration)
print(thzFreq/1e12)

print(pulseTime)

t = np.linspace(-pulseTime,pulseTime,2500)

pulse, env = gausspulse(t,fc=thzFreq,retenv=True)

t2 = np.linspace(0,pulseTime*2,2500)

thz_trace = np.zeros(2500)
for i, point in enumerate(pulse):
    thz_trace[i] = point

# plt.figure(2)
# plt.plot(t2,thz_trace,t2,env,'--')
plt.show()