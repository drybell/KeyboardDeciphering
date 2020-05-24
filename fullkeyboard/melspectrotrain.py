# https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06
# FULL KEYBOARD EDITION 
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, Input, MaxPool2D)
from tensorflow.keras.models import Model

_FFT_SIZE = 1024
_HOP_SIZE = 512
_N_MEL_BINS = 64
_N_SPECTROGRAM_BINS = (_FFT_SIZE // 2) + 1
_F_MIN = 0.0
_SAMPLE_RATE = 44100
_F_MAX = _SAMPLE_RATE / 2

class LogMelSpectrogram(tf.keras.layers.Layer):
    """Compute log-magnitude mel-scaled spectrograms."""

    def __init__(self, sample_rate, fft_size, hop_size, n_mels,
                 f_min=0.0, f_max=None, **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max)

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LogMelSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq, ch)
            The corresponding batch of log-mel-spectrograms
        """
        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        def power_to_db(magnitude, amin=1e-16, top_db=80.0):
            """
            https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
            """
            ref_value = tf.reduce_max(magnitude)
            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

            return log_spec

        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.fft_size,
                                      frame_step=self.hop_size,
                                      pad_end=False)

        magnitude_spectrograms = tf.abs(spectrograms)

        mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                     self.mel_filterbank)

        log_mel_spectrograms = power_to_db(mel_spectrograms)

        # add channel dimension
        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, 3)

        return log_mel_spectrograms

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'n_mels': self.n_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(LogMelSpectrogram, self).get_config())

        return config


def ConvModel(n_classes, sample_rate=44100, duration=.3,
              fft_size=_FFT_SIZE, hop_size=_HOP_SIZE, n_mels=_N_MEL_BINS):
    n_samples = sample_rate * duration
    
    # Accept raw audio data as input
    x = Input(shape=(int(n_samples),), name='input', dtype='float32')
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


AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_dataset(df):
    file_path_ds = tf.data.Dataset.from_tensor_slices(df.file_path)
    label_ds = tf.data.Dataset.from_tensor_slices(df.label)
    return tf.data.Dataset.zip((file_path_ds, label_ds))


def load_audio(file_path, label):
    # Load one second of audio at 44.1kHz sample-rate
    audio = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=13230)
    audio = tf.squeeze(audio)
    return audio, label


def prepare_for_training(ds, shuffle_buffer_size=64, batch_size=32):
    # Randomly shuffle (file_path, label) dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Load and decode audio from file paths
    ds = ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    # Repeat dataset forever
    ds = ds.repeat()
    # ds = tf.squeeze(ds)
    # Prepare batches

    # Prefetch

    # ds = tf.shape(ds)[0]
    # ds = tf.data.Dataset.unbatch(ds)
    # ds = tf.slice(ds, [0,0,0], [0, 1, 1])

    ###print(ds)
    ###<PrefetchDataset shapes: ((None, 44100, 1), (None,)), types: (tf.float32, tf.int32)>
    return ds

def prepare_for_testing(ds, shuffle_buffer_size=64, batch_size=32):
    # Randomly shuffle (file_path, label) dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Load and decode audio from file paths
    ds = ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds



def main():
    # Load meta.csv containing file-paths and labels as pd.DataFrame
    df = pd.read_csv('meta.csv')
    df2 = pd.read_csv('meta2.csv')
    
    ds = get_dataset(df)
    ds2 = get_dataset(df2)

    train_ds = prepare_for_training(ds)
    validate_ds = prepare_for_training(ds2)

    batch_size = 32
    train_steps = 32

	# For the first trial we will do the top row of the keyboard, 
	# which will be a total of 10 classes 
    model = ConvModel(25)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['sparse_categorical_accuracy'])
    model.summary()
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.BatchNormalization(momentum=0.98,input_shape=(44100, 1)))
    # # model.add(tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(128, return_sequences = True)))
    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    # opt = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
    # model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=['accuracy'])
    # # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


    # need to add a 28 output layer signifying a-z with spacebar and backspace
    # checkpoint_path = "training_1/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)

    print(train_ds)

    history = model.fit(train_ds, epochs=50, steps_per_epoch=train_steps, use_multiprocessing=True, validation_data=validate_ds, validation_steps=32)
    results = model.evaluate(validate_ds, batch_size=None, steps=32)
    print(history.history)
    print(results)
    model.save("./fullkeyboard.h5", include_optimizer=True)


if __name__ == '__main__':
    main()