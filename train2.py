import pandas as pd
import tensorflow as tf

# TRY JUST SPACEBAR AND BACKSPACE 

# FIGURE OUT HOW TO CUT DOWN ON RAM USAGE 
#       BOOST CPU USAGE 

# PC TENSORFLOW STILL BROKEN 
# MAC TENSORFLOW STILL USELESS 

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_dataset(df):
    file_path_ds = tf.data.Dataset.from_tensor_slices(df.file_path)
    label_ds = tf.data.Dataset.from_tensor_slices(df.label)
    return tf.data.Dataset.zip((file_path_ds, label_ds))


def load_audio(file_path, label):
    # Load one second of audio at 44.1kHz sample-rate
    audio = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio,
                                             desired_channels=1,
                                             desired_samples=44100)
    return audio, label


def prepare_for_training(ds, shuffle_buffer_size=1024, batch_size=64):
    # Randomly shuffle (file_path, label) dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Load and decode audio from file paths
    ds = ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    # Repeat dataset forever
    ds = ds.repeat()
    # Prepare batches
    ds = ds.batch(batch_size)
    # Prefetch
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def main():
    # Load meta.csv containing file-paths and labels as pd.DataFrame
    df = pd.read_csv('meta.csv')
    
    ds = get_dataset(df)
    train_ds = prepare_for_training(ds)

    batch_size = 16
    train_steps = 6

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization(momentum=0.98,input_shape=(44100, 1)))
    # model.add(tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(128, return_sequences = True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    opt = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
    model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


    # need to add a 28 output layer signifying a-z with spacebar and backspace
    checkpoint_path = "training_1/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(train_ds, epochs=50, steps_per_epoch=train_steps, use_multiprocessing=True, callback=[cp_callback] )


if __name__ == '__main__':
    main()