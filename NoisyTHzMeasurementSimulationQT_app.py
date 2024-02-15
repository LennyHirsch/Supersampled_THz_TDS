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
        self.padInput = QLineEdit()
        self.padInput.setText('100')

        self.samplesInput = QLineEdit()
        self.samplesInput.setText('100')

        self.amplitudeNoiseInput = QLineEdit()
        self.amplitudeNoiseInput.setText('0')

        self.timeJitterInput = QLineEdit()
        self.timeJitterInput.setText('0')

        self.probeFreqInput = QLineEdit()
        self.probeFreqInput.setText('5000')

        self.thzFreqInput = QLineEdit()
        self.thzFreqInput.setText('5')

        self.samplingFreqInput = QLineEdit()
        self.samplingFreqInput.setText('1000')

        #GRAPHS
        self.figure = pg.PlotWidget()
        self.figure.setClipToView(True)
        self.figure.setClipToView(True)
        self.figure.setLabel('left', text='Signal', units='a.u.')
        self.figure.setLabel('bottom', text='Time', units='a.u.')

        #LABELS
        self.padLabel = QLabel('Padding:')
        self.samplesLabel = QLabel('Samples:')
        self.amplitudeNoiseLabel = QLabel('Amplitude noise:')
        self.timeJitterLabel = QLabel('Time jitter:')
        self.probeFreqLabel = QLabel('Probe freq:')
        self.thzFreqLabel = QLabel('THz freq:')
        self.samplingFreqLabel = QLabel('Sampling freq:')


        #PROCESS RAW DATA PAGE
        layoutGenerate = QHBoxLayout()
        layoutGenerate.addWidget(self.probeFreqLabel)
        layoutGenerate.addWidget(self.probeFreqInput)
        layoutGenerate.addWidget(self.thzFreqLabel)
        layoutGenerate.addWidget(self.thzFreqInput)
        layoutGenerate.addWidget(self.samplingFreqLabel)
        layoutGenerate.addWidget(self.samplingFreqInput)
        layoutGenerate.addWidget(self.padLabel)
        layoutGenerate.addWidget(self.padInput)
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
        samplingFreq = int(self.samplingFreqInput.text())
        probeFreq = int(self.probeFreqInput.text())
        thzFreq = int(self.thzFreqInput.text())

        t_probe = np.linspace(-0.1,0.1,samplingFreq)
        p,probe = pulse(t_probe,probeFreq,retenv=True)

        t_thz = np.linspace(-1,1,samplingFreq)
        t,thz = pulse(t_thz,thzFreq,retenv=True)

        thz_trace = np.zeros(samplingFreq)
        for i, point in enumerate(t):
            thz_trace[i] = point

        probePulse = np.ndarray.tolist(probe)
        probePulse = [p for p in probe if p > 0.01]
        thzPulse = np.ndarray.tolist(thz_trace)

        self.probePulse = probePulse
        self.thzPulse = thzPulse
        self.thzLen = len(thzPulse)

        pad = int(self.padInput.text())
        print(f"THz trace length before padding: {self.thzLen}")
        for i in range(pad):
            self.thzPulse.insert(0,0)
            self.thzPulse.append(0)
        print(f"THz trace length after padding: {len(self.thzPulse)}")

    def noisyConvolution(self):
        probe = self.probePulse
        thz = self.thzPulse
        samples = int(self.samplesInput.text())
        pad = int(self.padInput.text())
        amplNoise = float(self.amplitudeNoiseInput.text())
        timeNoise = float(self.timeJitterInput.text())

        self.convolved = []
        self.std = []
        print(f"Padding = {pad}")
        print(f"Probe length = {len(probe)}")
        startIndex = int(pad - np.floor(len(probe)/2))
        # thzLen = len(thz)
        random.seed(time.time())

        for i in range(self.thzLen):
            currentValue = []
            for n in range(samples):
                dA = random.uniform(1-amplNoise, 1+amplNoise) #without amplitude noise, multiplier should be 1, so dA is centered around 1
                dt = int(random.uniform(0-timeNoise, 0+timeNoise)) #without time jitter, dt should be 0, so dt centered around 0
                noisyProbe = [p*dA for p in probe]
                currentIndex = startIndex + i + dt
                currentValue.append(np.dot(noisyProbe, thz[currentIndex:currentIndex+len(noisyProbe)]))
            self.convolved.append(np.average(currentValue))
            self.std.append(np.std(currentValue))

        print('noisyConvolution() finished')
        print(len(self.convolved))
        
        self.figure.plot(self.probePulse, pen=(255,255,255))
        self.figure.plot(self.convolved, pen=hsv2rgb(self.numOfPlots/10,1,1))
        self.figure.plot(self.std, pen=hsv2rgb(self.numOfPlots/10,1,0.5))
        self.numOfPlots += 1

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