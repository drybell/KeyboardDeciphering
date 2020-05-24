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

directory = "./spacebackspace/"

# files1 = os.listdir(directories[0])
# files2 = os.listdir(directories[1])
# files3 = os.listdir(directories[2])
# files4 = os.listdir(directories[3])
# files5 = os.listdir(directories[4])

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

# files1.sort(key=natural_keys)
# files2.sort(key=natural_keys)
# files3.sort(key=natural_keys)
# files4.sort(key=natural_keys)
# files5.sort(key=natural_keys)

files = os.listdir(directory)
files.sort(key=natural_keys)

with open("meta.csv", "w") as f: 
	f.write("file_path,label\n")
	for file in files: 
		# print("Current Directory is: " + directories[i])
		filename = file.split(".")[0]
		filename = re.sub("\d", "", filename)
		# print(filename)
		if filename == "backspace":
			filename = "0"
		elif filename == "space": 
			filename = "1"
		else:
			filename = str(ord(filename))
		f.write(directory + file + ", " + filename + "\n")

# for file in files:
# 	rate, data = sio.read(directory + file)
# 	print(data)
# 	audio_binary = tf.read_file(directory + filename)
# 	desired_channels = 1
# 	wav_decoder = contrib_audio.decode_wav(audio_binary, desired_channels=desired_channels)

