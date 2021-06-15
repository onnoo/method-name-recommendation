
### data path ###
TRAIN_DATA = 'data/training-med.txt'
VALID_DATA = 'data/validation-med.txt'
TEST_DATA = 'data/test-med.txt'
MODEL_CHECKPOINT = 'checkpoint/best.ckpt'

### model hyper params ###
SOS_TOKEN = '\t'
EOS_TOKEN = '\n'
NUM_WORDS = 20000
MAX_LEN_INPUT = 85
MAX_LEN_TARGET = 9
EMBEDDING_DIM = 400
LATENT_DIM = 512

### training hyper params ###
BATCH_SIZE = 256
EPOCHS = 50
