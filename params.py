
### data path ###
TRAIN_DATA = 'data/training.txt'
VALID_DATA = 'data/validation.txt'
TEST_DATA = 'data/test.txt'
MODEL_CHECKPOINT = 'best.ckpt'

### model hyper params ###
SOS_TOKEN = '\t'
EOS_TOKEN = '\n'
NUM_WORDS = 20000
MAX_LEN_INPUT = 85
MAX_LEN_TARGET = 9
EMBEDDING_DIM = 100
LATENT_DIM = 512

### training hyper params ###
BATCH_SIZE = 64
EPOCHS = 10