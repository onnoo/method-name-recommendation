import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import argparse
import params
import text
from model import MNire


if __name__ == '__main__':

    ### load data
    x_test, y_test = text.prepare_text(params.TEST_DATA)

    with open('tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)
    
    word2idx = tokenizer.word_index
    print(f'Found {len(word2idx)} unique input tokens.')

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=params.MAX_LEN_INPUT)

    model = MNire(word2idx[params.SOS_TOKEN], word2idx[params.EOS_TOKEN])
    model.load_weights(params.MODEL_CHECKPOINT)


    results = []

    for idx, (x, y) in enumerate(zip(x_test, y_test)):
        pred = model(x_test[idx:idx+1]).numpy()
        pred = tokenizer.sequences_to_texts(pred)[0].rstrip()
        
        result = f'{y},{pred}\n'
        results.append(result)

        if idx > 1000:
            break
    
    with open('results.txt', 'w') as f:
        f.writelines(results)
