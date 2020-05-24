import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import csv
from pydub import AudioSegment 
import os 

directory = "./keyboardfinaltay/test10"
sound = "keyboard10.wav"
text  = "intemperant.csv"

sound_dir = directory + "/" + sound
text_dir = directory + "/" + text

file = open(text_dir, "r") 

words = []
time = []
total_times = []

total_time = 0

data = csv.reader(file, delimiter=",")

starter_word = "`"
flag = False
counter = 0
for row in data:
	try:
		if row[0] == starter_word and flag == False: 
			flag = True
		elif flag == True: 
			# print("KEY: " + row[0] + ", TIME: " + row[1])
			words.append(row[0])
			time.append(row[1])
			total_time += float(row[1])
			total_times.append(total_time)
		else: 
			counter += 1
			pass
	except IndexError: 
		break

print("Num of keys pressed before `: " + str(counter))

#Don't count the ` key dt 
numbers_skipped = counter + 1

# print(total_time)
# print(total_times)

spf = wave.open(sound_dir, "r")

# Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, "Int16")
fs = 44100

# If Stereo
if spf.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)

Time = np.linspace(0, len(signal) / fs, num=len(signal))


wf = AudioSegment.from_wav(sound_dir)
### NAME PROTOCOL FOR THE SMALL WAV FILES: LETTER TYPED + POSITION INCREMENT 
os.mkdir("./keyboardfinal/validatetay2")
counter = 0
plt.figure(1)
plt.title("Signal Wave...")
plt.plot(Time, signal, linewidth=.4, color='black')
for time in total_times: 
	plt.axvline(x=time, ymin=-15000, ymax=15000, color="red", linewidth = .1)
	if counter == len(total_times) - 1: 
		pass
	else: 
		outputfile = words[counter] + str(counter) + ".wav"
		t1 = time * 1000 
		t2 = (total_times[counter + 1] - .02) * 1000 
		wf1 = wf[t1:t2]
		wf1.export("keyboardfinal/validatetay2/" + outputfile, format='wav')
		counter += 1 
plt.show()

spf.close()
file.close()

