SAVING_PATH = 'model/' # path to save the model
MODEL_PATH = 'model/model_best.pth' # path to save the model
TRAINING_PATH = 'model/train.tsv' # path to load the training data
DEV_PATH = 'model/dev.tsv' # path to load the validation data
TEST_PATH = '' # path to load the test data
UNK_WORD = '<UNK>' # unk token
PAD_TOKEN = '<PAD>' # pad token
PRETRAINED_PATH = 'model/glove_pretrained_300'

# Assign each tag with a unique index 
TAG_DICT = { "B-CORP" : 0, "B-CW" : 1, "B-GRP" : 2, "B-LOC" : 3,
             "B-PER" : 4, "B-PROD" : 5, "I-CORP" : 6, "I-CW" : 7,
             "I-GRP" : 8, "I-LOC" : 9, "I-PER" : 10, "I-PROD" : 11, "O" : 12 } 

# weights for the loss function
WEIGHTS = None

# Hyperparameters
LR = 0.35 # learning rate
EPOCHS = 50 # number of epochs
BATCH_SIZE = 32 # batch size
MOMENTUM = 0.94 # momentum
WEIGHT_DECAY = 0.00008 # weight decay

EMBEDDING_SIZE = 300 # size of embedding
HIDDEN_SIZE = 100 # size of hidden layer of LSTM
NUM_LAYERS = 2  # number of layers of LSTM
BIDIRECTIONAL = True # bidirectional LSTM
PRETRAINED = True # use pretrained embeddings
