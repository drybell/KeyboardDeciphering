import tensorflow as tf 

def ConvModel(n_classes, sample_rate=44100, duration=1,
              fft_size=_FFT_SIZE, hop_size=_HOP_SIZE, n_mels=_N_MEL_BINS):
    n_samples = sample_rate * duration
    
    # Accept raw audio data as input
    x = Input(shape=(n_samples,), name='input', dtype='float32')
    # Process into log-mel-spectrograms. (This is your custom layer!)
    y = LogMelSpectrogram(sample_rate, fft_size, hop_size, n_mels)(x)
    # Normalize data (on frequency axis)
    y = BatchNormalization(axis=2)(y)
    
    y = Conv2D(32, (3, n_mels), activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPool2D((1, y.shape[2]))(y)

    y = Conv2D(32, (3, 1), activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPool2D(pool_size=(2, 1))(y)

    y = Flatten()(y)
    y = Dense(64, activation='relu')(y)
    y = Dropout(0.25)(y)
    y = Dense(n_classes, activation='softmax')(y)

    return Model(inputs=x, outputs=y)

# model = ConvModel(10)
model = tf.load_weights("toprow.h5")
model.