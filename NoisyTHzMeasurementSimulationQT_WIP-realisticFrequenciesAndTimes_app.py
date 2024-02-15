from PyQt5.QtCore import Qt
from pathlib import Path
import pyqtgraph as pg
import pandas as pd
import numpy as np
import random
from customUtilities import hsv2rgb
import time
from scipy.signal import gausspulse as pulse
from scipy.fft import rfft, rfftfreq
import sys # We need sys so that we can pass argv to QApplication
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QWidget,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Noisy THz convolution simulation")

        self.thzPulse = []
        self.probePulse = []
        self.thzLen = 0
        self.convolved = []
        self.std = []
        self.numOfPlots = 0

        self.samplingFreq = int(np.ceil((1.25e-12 / 1e-15)*2))
        self.stepSizeDistance = 3e-7

        #BUTTONS
        self.generateButton = QPushButton('Generate pulses')
        self.generateButton.clicked.connect(self.generatePulses)

        self.convolveButton = QPushButton('Convolve')
        self.convolveButton.clicked.connect(self.noisyConvolution)

        self.clearPlotsButton = QPushButton('Clear plots')
        self.clearPlotsButton.clicked.connect(self.clearPlots)

        self.clearDataButton = QPushButton('Clear data')
        self.clearDataButton.clicked.connect(self.clearData)

        #INPUTS
        
        self.probeDurationInput = QLineEdit()
        self.probeDurationInput.setText('250e-15')

        self.thzFreqInput = QLineEdit()
        self.thzFreqInput.setText('2.5e12')

        self.totalTimeInput = QLineEdit()
        self.totalTimeInput.setText('1.25e-12')
        self.totalTimeInput.textChanged.connect(self.updateSamplingFreq)

        self.samplingTimeInput = QLineEdit()
        self.samplingTimeInput.setText('1')
        self.samplingTimeInput.textChanged.connect(self.updateSamplingFreq)

        # self.samplingFreqInput = QLineEdit()
        # self.samplingFreqInput.setText('2500')

        self.probeSamplesInput = QLineEdit()
        self.probeSamplesInput.setText('100')

        self.samplesInput = QLineEdit()
        self.samplesInput.setText('100')

        self.amplitudeNoiseInput = QLineEdit()
        self.amplitudeNoiseInput.setText('0')

        self.timeJitterInput = QLineEdit()
        self.timeJitterInput.setText('0')

        self.stepSizeInput = QLineEdit()
        self.stepSizeInput.setText('1')
        self.stepSizeInput.textChanged.connect(self.updateStepSize)

        #GRAPHS
        self.figure = pg.PlotWidget()
        self.figure.setClipToView(True)
        self.figure.setClipToView(True)
        self.figure.setLabel('left', text='Signal', units='a.u.')
        self.figure.setLabel('bottom', text='Time', units='s')

        #LABELS
        self.probeSamplesLabel = QLabel('Probe samples:')
        self.samplesLabel = QLabel('Samples:')
        self.amplitudeNoiseLabel = QLabel('Amplitude noise:')
        self.timeJitterLabel = QLabel('Time jitter:')
        self.probeDurationLabel = QLabel('Probe duration (s):')
        self.thzFreqLabel = QLabel('THz freq (Hz):')
        self.totalTimeLabel = QLabel('Total time (s):')
        self.samplingTimeLabel = QLabel('Sample time (fs):')
        self.samplingFreqLabel = QLabel(f"Number of samples: {self.samplingFreq}")
        self.stepSizeLabel = QLabel('Step size (fs):')
        self.stepSizeDistanceLabel = QLabel(f"= {self.stepSizeDistance*10**6:.1f}μm")

        #PROCESS RAW DATA PAGE
        layoutGenerate = QHBoxLayout()
        layoutGenerate.addWidget(self.probeDurationLabel)
        layoutGenerate.addWidget(self.probeDurationInput)
        layoutGenerate.addWidget(self.probeSamplesLabel)
        layoutGenerate.addWidget(self.probeSamplesInput)
        layoutGenerate.addWidget(self.thzFreqLabel)
        layoutGenerate.addWidget(self.thzFreqInput)
        layoutGenerate.addWidget(self.totalTimeLabel)
        layoutGenerate.addWidget(self.totalTimeInput)
        layoutGenerate.addWidget(self.samplingTimeLabel)
        layoutGenerate.addWidget(self.samplingTimeInput)
        layoutGenerate.addWidget(self.samplingFreqLabel)
        layoutGenerate.addWidget(self.generateButton)
        
        loadGenerateWidgets = QWidget()
        loadGenerateWidgets.setLayout(layoutGenerate)

        #LOAD AND PLOT PROCESSED DATA PAGE
        layoutConvolve = QHBoxLayout()
        layoutConvolve.addWidget(self.samplesLabel)
        layoutConvolve.addWidget(self.samplesInput)
        layoutConvolve.addWidget(self.amplitudeNoiseLabel)
        layoutConvolve.addWidget(self.amplitudeNoiseInput)
        layoutConvolve.addWidget(self.timeJitterLabel)
        layoutConvolve.addWidget(self.timeJitterInput)
        layoutConvolve.addWidget(self.stepSizeLabel)
        layoutConvolve.addWidget(self.stepSizeInput)
        layoutConvolve.addWidget(self.stepSizeDistanceLabel)
        layoutConvolve.addWidget(self.convolveButton)

        loadConvolveWidgets = QWidget()
        loadConvolveWidgets.setLayout(layoutConvolve)

        layoutClear = QHBoxLayout()
        layoutClear.addWidget(self.clearDataButton)
        layoutClear.addWidget(self.clearPlotsButton)

        loadClearWidgets = QWidget()
        loadClearWidgets.setLayout(layoutClear)

        layoutPlot = QVBoxLayout()
        layoutPlot.addWidget(loadGenerateWidgets)
        layoutPlot.addWidget(loadConvolveWidgets)
        layoutPlot.addWidget(self.figure)
        layoutPlot.addWidget(loadClearWidgets)

        plotDataWidgets = QWidget()
        plotDataWidgets.setLayout(layoutPlot)

        self.setCentralWidget(plotDataWidgets)

    def generatePulses(self):
        self.thzPulse = []
        self.probePulse = []
        self.thzLen = 0
        self.convolved = []
        self.std = []

        samplingFreq = self.samplingFreq
        probeDuration = float(self.probeDurationInput.text())
        thzFreq = float(self.thzFreqInput.text())
        totalTime = float(self.totalTimeInput.text())

        print(f"Probe duration = {self.probeDurationInput.text()}s")

        self.probeTime = np.linspace(0, probeDuration*4, int(self.probeSamplesInput.text()))
        amplitude = 85e-6 #85uW
        center = probeDuration*2
        width = probeDuration
        for x in self.probeTime: self.probePulse.append(amplitude*np.exp(-((x-center)**2)/(0.5*width**2)))

        t_thz = np.linspace(-totalTime,totalTime,samplingFreq)
        t,thz = pulse(t_thz,thzFreq,retenv=True)

        self.t2 = np.linspace(0,totalTime*2,samplingFreq)


        thz_amplitude = 35e-6
        thz_trace = np.zeros(samplingFreq)
        for i, point in enumerate(t):
            thz_trace[i] = point * thz_amplitude

        thzPulse = np.ndarray.tolist(thz_trace)

        print(f"Probe length = {len(self.probePulse)}")

        # self.probePulse = probePulse
        self.thzPulse = thzPulse
        self.thzLen = len(thzPulse)

        pad = int(self.probeSamplesInput.text())
        print(f"THz trace length before padding: {self.thzLen}")
        for i in range(pad):
            self.thzPulse.insert(0,0)
            self.thzPulse.append(0)
        print(f"THz trace length after padding: {len(self.thzPulse)}")

    def noisyConvolution(self):
        probe = self.probePulse
        thz = self.thzPulse
        samples = int(self.samplesInput.text())
        pad = int(self.probeSamplesInput.text())
        amplNoise = float(self.amplitudeNoiseInput.text())
        timeNoise = float(self.timeJitterInput.text())
        stepSize = int(self.stepSizeInput.text())

        self.convolved = []
        self.std = []
        print(f"Padding = {pad}")
        print(f"Probe length = {len(probe)}")
        startIndex = int(pad - np.floor(len(probe)/2))
        # thzLen = len(thz)
        random.seed(time.time())

        for i in range(int(self.thzLen/stepSize)):
            currentValue = []
            for n in range(samples):
                dA = random.uniform(1-amplNoise, 1+amplNoise) #without amplitude noise, multiplier should be 1, so dA is centered around 1
                dt = int(random.uniform(0-timeNoise, 0+timeNoise)) #without time jitter, dt should be 0, so dt centered around 0
                noisyProbe = [p*dA for p in probe]
                currentIndex = startIndex + i*stepSize + dt
                currentValue.append(np.dot(noisyProbe, thz[currentIndex:currentIndex+len(noisyProbe)])*300) #x300 to scale THz trace up to approx original THz level
            self.convolved.append(np.average(currentValue))
            self.std.append(np.std(currentValue))

        print('noisyConvolution() finished')
        maxConvolved = 0
        if self.numOfPlots == 0:
            maxConvolved = max(self.convolved)
        
        t3 = np.linspace(0,float(self.totalTimeInput.text())*2,int(self.samplingFreq/stepSize))

        self.figure.plot(self.probeTime, self.probePulse, pen=hsv2rgb(0,0,0.5))
        # self.figure.plot(self.t2, [s*maxConvolved for s in self.thzPulse[pad:-pad]], pen=hsv2rgb(0,0,0.5)) #plot original pulse
        self.figure.plot(self.t2, [s for s in self.thzPulse[pad:-pad]], pen=hsv2rgb(0,0,0.5)) #plot original pulse
        self.figure.plot(t3, self.convolved, pen=hsv2rgb(self.numOfPlots/10,1,1))
        self.figure.plot(t3, self.std, pen=hsv2rgb(self.numOfPlots/10,1,0.5))
        self.numOfPlots += 1

    def updateSamplingFreq(self):
        self.samplingFreq = int(np.ceil(2*(float(self.totalTimeInput.text())/(float(self.samplingTimeInput.text())*10**-15))))
        self.samplingFreqLabel.setText(f"Number of samples: {self.samplingFreq}")

    def updateStepSize(self):
        self.stepSizeDistance = float(3e8 * int(self.stepSizeInput.text())*10**-15)
        self.stepSizeDistanceLabel.setText(f"= {self.stepSizeDistance*10**6:.1f}μm")

    def clearPlots(self):
        self.figure.clear()
        self.numOfPlots = 0

    def clearData(self):
        self.thzPulse = []
        self.probePulse = []
        self.thzLen = 0
        self.convolved = []
        self.std = []

app = QApplication(sys.argv)
# app.setStyle('Windows')
window = MainWindow()
window.show()

app.exec()