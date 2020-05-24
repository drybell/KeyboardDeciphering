import scipy.io.wavfile as sio 
import os 
import re
# import numpy as np 
# import tensorflow as tf 
# from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

#### DATA 
#    sound as numpy array   |   letter that is pressed   |  length of audio 


### maybe pad audio sequences so all are the longest in the batch? 
### or use length of audio as a data point 

directories = ["./keyboardfinal/taytest1/","./keyboardfinal/taytest2/","./keyboardfinal/taytest3/",
								"./keyboardfinal/taytest4/", "./keyboardfinal/taytest5/", "./keyboardfinal/taytest6/",
								"./keyboardfinal/taytest7/", "./keyboardfinal/taytest8/", "./keyboardfinal/validatetay1/",
								"./keyboardfinal/validatetay2/"]

files1 = os.listdir(directories[0])
files2 = os.listdir(directories[1])
files3 = os.listdir(directories[2])
files4 = os.listdir(directories[3])
files5 = os.listdir(directories[4])
files6 = os.listdir(directories[5])
files7 = os.listdir(directories[6])
files8 = os.listdir(directories[7])
files9 = os.listdir(directories[8])
files10 = os.listdir(directories[9])

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

files1.sort(key=natural_keys)
files2.sort(key=natural_keys)
files3.sort(key=natural_keys)
files4.sort(key=natural_keys)
files5.sort(key=natural_keys)
files6.sort(key=natural_keys)
files7.sort(key=natural_keys)
files8.sort(key=natural_keys)
files9.sort(key=natural_keys)
files10.sort(key=natural_keys)

fileslist = [files1, files2, files3, files4, files5, files6, files7, files8, files9, files10]

# files = os.listdir(directory)
# files.sort(key=natural_keys)

with open("metatay.csv", "w") as f: 
	f.write("file_path,label\n")
	for i in range(0, 10):
		for file in fileslist[i]: 
			# print("Current Directory is: " + directories[i])
			# print(file)
			filename = file.split(".")[0]
			filename = re.sub("\d", "", filename)
			# print(filename)
			if filename == "backspace":
				filename = "0"
			elif filename == "space": 
				filename = "1"
			else:
				filename = str(ord(filename) - 95)
			f.write(directories[i] + file + ", " + filename + "\n")

# for file in files:
# 	rate, data = sio.read(directory + file)
# 	print(data)
# 	audio_binary = tf.read_file(directory + filename)
# 	desired_channels = 1
# 	wav_decoder = contrib_audio.decode_wav(audio_binary, desired_channels=desired_channels)

