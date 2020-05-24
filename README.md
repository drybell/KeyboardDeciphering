# Instruments Final Spring 2020

For my instruments final project my teammates Harris Hardiman-Mostow, Taylor Korte, and I
gathered data, tested, and trained a model to be able to recover keypresses from 
audio data of us typing. 

We initially began with training 200 spacebar and backspace key presses from each team member
which gave us 86% validation accuracy using just the raw audio, no preprocessing. 

We then started using MelSpectrogram Analysis as a custom layer in our model when using more than
2 classes. We ended up using 28 different tags: a-z and backspace/spacebar for final prediction. 

Our final model, which included Taylor and my combined dataset of 2000 random key presses, was able to 
train to 98% accuracy and 65% validation accuracy.

Once I restructure the code and make it easier to ship, hopefully getting more data from different 
computers could help increase our validation accuracy and further support audio keypress recovery
or "keyboard decipherment" as our professor called it. 

# How to Use

	TODO: need to write the dependency file for shipping 
	python3 typer.py -w outputwavfile.wav
		press ` and type the words on the screen 

	This will output a csv and wav file. The csv file contains the letters that were pressed, and 
	the wav file is the audio from when the user presses the ` key

	Running python3 plotwav.py and modifying the directory + filename gives us the individual trimmed
	audio files with the filenames representing the keypress. Navigate to keyboardfinal/train1 to see 
	what plotwav outputs. 

	Afterwards, modify the meta.csv file using makemeta.py, run python3 melspectrotrain.py allows us to train the model. 

	Our last model was fullkeyboardboth.h5 using data from keyboardfinal 

	This is pretty messy due to time constraints, but hopefully I can condense the workflow to just 
	running typer.py which automatically trims and updates the meta.csv file so I can immediately
	run melspectrotrain 


# Helpful links for this project 

I adapted a lot of code from 2 medium blog posts about similar topics 
https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc
https://towardsdatascience.com/how-to-build-efficient-audio-data-pipelines-with-tensorflow-2-0-b3133474c3c1

