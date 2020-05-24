import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import wave
import sys
import csv
from pydub import AudioSegment 
import os 
import re 

directory = "test1/dan2_1/"

rows = 10
columns = 15

fig, axs = plt.subplots(rows, columns)
axs = axs.flatten()

files = os.listdir(directory) 

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

files.sort(key=natural_keys)

for file, ax in zip(files, axs):
	with wave.open(directory + file, "r") as spf:
		filename = file.split(".")[0]
		filename = re.sub("\d", "", filename)

		signal = spf.readframes(-1)
		signal = np.fromstring(signal, "Int16")
		fs = 44100

		Time = np.linspace(0, len(signal) / fs, num=len(signal))

		ax.set_title(filename, fontsize=6)
		ax.set_axis_off()
		ax.plot(Time, signal, linewidth=.4, color="black")

plt.show()
plt.axis('off')
