import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import callbacks

import argparse
import params
import text
from model import MNire


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('skip', default=False, type=bool)
    args = parser.parse_args()

    ### load data
    x_train, y_train = text.prepare_text(params.TRAIN_DATA)
    x_valid, y_valid = text.prepare_text(params.VALID_DATA)

    encoder_input_train = x_train
    decoder_input_train = [params.SOS_TOKEN + ' ' + t for t in y_train]
    decoder_target_train = [t + ' ' + params.EOS_TOKEN for t in y_train]

    encoder_input_valid = x_valid
    decoder_input_valid = [params.SOS_TOKEN + ' ' + t for t in y_valid]
    decoder_target_valid = [t + ' ' + params.EOS_TOKEN for t in y_valid]


    ### tokenize texts
    tokenizer = Tokenizer(num_words=params.NUM_WORDS, filters='!"#$%&()*+-/.,<=>?@[\\]^_`{|}~')
    tokenizer.fit_on_texts(encoder_input_train + decoder_input_train + decoder_target_train)

    with open('tokenizer.pickle', 'wb') as file:
        pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    word2idx = tokenizer.word_index
    print(f'Found {len(word2idx)} unique input tokens.')

    encoder_input_train = tokenizer.texts_to_sequences(encoder_input_train)
    decoder_input_train = tokenizer.texts_to_sequences(decoder_input_train)
    decoder_target_train = tokenizer.texts_to_sequences(decoder_target_train)

    encoder_input_valid = tokenizer.texts_to_sequences(encoder_input_valid)
    decoder_input_valid = tokenizer.texts_to_sequences(decoder_input_valid)
    decoder_target_valid = tokenizer.texts_to_sequences(decoder_target_valid)
    

    ### padding sequences
    encoder_input_train = pad_sequences(encoder_input_train, maxlen=params.MAX_LEN_INPUT)
    decoder_input_train = pad_sequences(decoder_input_train, maxlen=params.MAX_LEN_TARGET, padding='post')
    decoder_target_train = pad_sequences(decoder_target_train, maxlen=params.MAX_LEN_TARGET, padding='post')

    encoder_input_valid = pad_sequences(encoder_input_valid, maxlen=params.MAX_LEN_INPUT)
    decoder_input_valid = pad_sequences(decoder_input_valid, maxlen=params.MAX_LEN_TARGET, padding='post')
    decoder_target_valid = pad_sequences(decoder_target_valid, maxlen=params.MAX_LEN_TARGET, padding='post')

    
    ### training
    model = MNire(word2idx[params.SOS_TOKEN], word2idx[params.EOS_TOKEN])
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    cp_callback = callbacks.ModelCheckpoint(filepath=params.MODEL_CHECKPOINT, save_weights_only=True, verbose=1)
    es_callback = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 2)
    
    if not args.skip:
        history = model.fit(x=[encoder_input_train, decoder_input_train],
                            y=decoder_target_train,
                            validation_data=([encoder_input_valid, decoder_input_valid], decoder_target_valid),
                            batch_size=params.BATCH_SIZE,
                            epochs=params.EPOCHS,
                            callbacks=[cp_callback, es_callback])    
