import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention
from params import NUM_WORDS, EMBEDDING_DIM, LATENT_DIM, MAX_LEN_TARGET


class Encoder(tf.keras.Model):
    
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.embedding = Embedding(NUM_WORDS, EMBEDDING_DIM)
        self.lstm = LSTM(
            LATENT_DIM,
            return_sequences=True,
            return_state=True
        )
    
    def call(self, x):
        x = self.embedding(x)
        
        H, h, c = self.lstm(x)
        
        return H, h, c


class Decoder(tf.keras.Model):
    
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.embedding = Embedding(NUM_WORDS, EMBEDDING_DIM)
        self.lstm = LSTM(
            LATENT_DIM,
            return_sequences=True,
            return_state=True
        )

        self.attention = Attention()
        self.dense = Dense(NUM_WORDS, activation='softmax')
    
    def call(self, inputs):
        # x : target sequence
        # s0, c0 : initial LSTM state (from last state of encoder)
        # H : hidden states of encoder
        x, s0, c0, H = inputs
        x = self.embedding(x)
        
        S, h, c = self.lstm(x, initial_state=[s0, c0])
        
        # use s(t-1) for prediction s(t)
        S_ = tf.concat([s0[:, tf.newaxis, :], S[:, :-1, :]], axis=1)
        
        A = self.attention([S_, H])  # Query: S_, Key: H, Value: H
        
        y = tf.concat([S, A], axis=-1)
        
        return self.dense(y), h, c


class MNire(tf.keras.Model):
    
    def __init__(self, sos, eos):
        super(MNire, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        self.sos = sos  # sos idx
        self.eos = eos  # eos idx
        
    def call(self, inputs, training=False, mask=None):
        if training is True:
            x, y = inputs
            
            H, h, c = self.enc(x)
            
            y, _, _ = self.dec([y, h, c, H])
            
            return y
        
        else:
            # inference for one sample
            x = inputs
            H, h, c = self.enc(x)
            
            y = tf.convert_to_tensor(self.sos, dtype=tf.int32)
            y = tf.reshape(y, (1, 1))
            
            seq = tf.TensorArray(tf.int32, MAX_LEN_TARGET)
            
            for idx in tf.range(MAX_LEN_TARGET):
                y, h, c = self.dec([y, h, c, H])
                
                y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
                y = tf.reshape(y, (1, 1))
                
                seq = seq.write(idx, y)
                
                if y == self.eos:
                    break
            
            return tf.reshape(seq.stack(), (1, MAX_LEN_TARGET))


    def train_step(self, data):

        inputs, targets = data
        encoder_inputs, decoder_inputs = inputs

        with tf.GradientTape() as tape:
            predictions = self([encoder_inputs, decoder_inputs], training=True)
            loss = self.compiled_loss(targets, predictions)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):

        inputs, targets = data
        encoder_inputs, decoder_inputs = inputs

        predictions = self([encoder_inputs, decoder_inputs], training=True)

        self.compiled_loss(targets, predictions, regularization_losses=self.losses)

        self.compiled_metrics.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}