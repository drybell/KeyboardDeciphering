# Daniel Ryaboshapka
# Warm up for quick testing 
# 
# Arguments for 3 different length texts --style: -s (short), -m (medium), -l (long)

import argparse 
import threading
import random

import pyaudio
import wave


## Print text-to-write on screen


## Type, without pressing enter, the words on the screen (backspace acceptable)


## record when you first press enter 


def record(FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, outputfile): 
	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                frames_per_buffer=CHUNK)

	print("* recording")

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)

	print("* done recording")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(outputfile, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()



def openFile(filename, size): 
	total_dict = []
	with open(filename, "r") as f:
		for line in f: 
			total_dict.append(line)
	case = {"small": 15, "medium": 30, "large": 60}
	subset_size = case.get(size, "Invalid choice")

	random_indices = []

	i = 0 
	while i != subset_size: 
		check = random.randint(0, len(total_dict))
		if check not in random_indices: 
			random_indices.append(check)
			i += 1

	words = []
	for index in random_indices: 
		words.append(str.rstrip(total_dict[index]))
	# print(words)
	return words

def typing(outputfile):
	typed = input("Please type the following passage to the best of your ability.\n" + 
		     "Backspaces are allowed, press Enter when complete:\n")
	
	with open(outputfile, "a") as f:
		f.write("User output was: " + typed)

	

def main(): 
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	RECORD_SECONDS = 30 #should specify short, medium, long 
	WAVE_OUTPUT_FILENAME = "output.wav"


	### MODIFY THIS FILENAME PER TEST
	### change 1 to a 2.... etc 
	text_file_output = "training10.txt"

	###
	###

	input_filename = "words_alpha.txt"

	words = openFile(input_filename, "small")

	outputfile = words[0] + ".wav"
	print("Output File is: " + outputfile)

	textfile = open(text_file_output, "w")
	textfile.write("Input was: ")
	counter = 1
	for word in words: 
		if counter > 8: 
			print(word)
			textfile.write(word + " \n")
			counter = 0
		else:
			print(word, end=" ")
			textfile.write(word + " ")
		counter += 1

	textfile.write("\n")
	textfile.close()

	print("\n")
	input("Press Enter to begin: ")

	### thread 1: audio 
	x = threading.Thread(target=record, args=(FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, outputfile))
	x.start()

	### thread 2: type input 
	y = threading.Thread(target=typing, args=(text_file_output,))
	y.start()

if __name__ == '__main__':
	main()