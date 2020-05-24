import time 
import curses
import os
import pyaudio
import concurrent.futures
import threading 
import wave
import argparse 
from audio_test1 import openFile

parser = argparse.ArgumentParser(description='Help collect test data for us!')
parser.add_argument('-w', dest='writeTo', help='Write to this audio filename <.wav> extension', default='output.wav')
args = parser.parse_args()
if args.writeTo:
   outputfile = args.writeTo 

record = False 
stop_callback = False
frames = []
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
#RECORD_SECONDS = 30 #should specify short, medium, long 


p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    global stream, record, frames, stop_callback
    if record: 
        # data = stream.read(1024)
        frames.append(in_data)

    if stop_callback:
        callback_flag = pyaudio.paComplete
    else:
        callback_flag = pyaudio.paContinue

    return (in_data, callback_flag)

stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=callback)
# `: begin recording 
# enter: exit, stop recording, output files 

# def record(CHUNK, frames, stream):
#     data = stream.read(CHUNK)
#     frames.append(data) 
#     return frames 

def main(win):
    global record, stop_callback, frames, CHUNK, FORMAT, CHANNELS, RATE, outputfile, p, stream
    start = time.time()
    input_filename = "words_alpha.txt"
    words = openFile(input_filename, "small")
    outputword = ""
    i = 0
    for word in words: 
        if i == 0:
            outputword = word
        elif i % 4 == 0: 
            outputword = outputword + "  " + word + "\n"
        else:
            outputword = outputword + "  " + word
        i += 1
    total_delays = []
    all_keys = []
    win.nodelay(True)
    key=""
    key_strings = ""
    win.clear()                
    win.addstr(0,0, "Detected key: ")
    win.addstr(1,0, "Current Time: " + time.strftime("%H:%M:%S",time.localtime()))
    win.addstr(2,0, "Time Elapsed: ")
    recordingstring = ""
    flag = True
    while True:   
        try:               
            key = win.getkey()  
            if key == " ":
                key = "space"
                key_strings += " "
            elif key in ('KEY_BACKSPACE', '\b', '\x7f'):
                key = "backspace"
                key_strings = key_strings[:-1]
            elif key == "`" and flag:
                recordingstring = "** RECORDING **"
                record = True
                flag = False
                stream.start_stream()
            else: 
                key_strings += str(key)
            all_keys.append(key)
            curr = time.time()       
            diff = curr - start   
            start = time.time()
            total_delays.append(diff)
            win.clear()                
            win.addstr(0,0,"Detected key: ")
            win.addstr(key) 
            win.addstr(1,0, "Current Time: " + time.strftime("%H:%M:%S",time.localtime()))
            win.addstr(2,0, "Time Elapsed: " + str(diff) + " seconds")
            win.addstr(3,0, recordingstring)
            win.addstr(4,0, outputword)
            if ord(key) < 126 and ord(key) > 32:
                win.addstr(10,0, key_strings)
            if key == os.linesep:
                stop_callback = True
                endtime = time.time()
                stream.stop_stream()
                with open((words[0] + ".csv"),"w") as f:
                    i = 0
                    for key in all_keys:
                        if i == 0:
                            f.write(key + ",0\n")
                        f.write(key + ","+ str(total_delays[i]) + "\n")
                        i += 1
                stream.close()
                p.terminate()
                wf = wave.open(outputfile, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                break
        except Exception as e:
            #     record = False
            #     with concurrent.futures.ThreadPoolExecutor() as executor:
            #         future = executor.submit(record, (CHUNK, frames, stream))
            #         return_value = future.result()
            if not flag:
                record = True
            win.refresh()
            win.addstr(3,0, recordingstring)
            win.addstr(10,0, key_strings)
            win.addstr(1,0, "Current Time: " + time.strftime("%H:%M:%S",time.localtime()))
            pass         


curses.wrapper(main)

# def main():
#   starttime = time.localtime() 
#   stdscr = curses.initscr()
#   endtime = curses.wrapper(run_window(stdscr))
#   time_diff = endtime - starttime 
#   print("Time Elapsed: " + str(time_diff) + " seconds")

