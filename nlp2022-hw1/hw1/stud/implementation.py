import numpy as np
import torch

from typing import List, Tuple
from model import Model
from stud.ner_pretrain_noisycorpus import NERv1_PRE
from stud.ner_pretrain_stdbatch import NERv2
from stud.data_loading import *
from stud.vocabulary import Vocabulary
from gensim.models import KeyedVectors
from stud.configuration import *

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device) #RandomBaseline()

class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]

class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device):
        self.device = device
        self.dataset = DatasetNER(TRAINING_PATH, verbose=False)
        self.weights = self.init_weights(PRETRAINED)
        self.vocab = self.init_vocab(self.dataset, self.weights)
        self.tags = TAG_DICT
        self.emb_size = EMBEDDING_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.bidirectional = BIDIRECTIONAL
        self.num_classes = len(self.tags)
        #self.model = NERv2(len(self.vocab), self.emb_size, self.hidden_size, self.num_layers, self.num_classes, self.bidirectional, device, self.weights)
        self.model = NERv1_PRE(len(self.vocab), self.emb_size, self.hidden_size, self.num_layers, self.num_classes, self.bidirectional, device)
        print("Model loaded")

    def init_weights(self, pretrained = False):
        if pretrained:
            weights = KeyedVectors.load(PRETRAINED_PATH)
            #pad = np.zeros_like(weights.get_vector("the"))
            #weights.add_vectors(PAD_TOKEN, pad)
            print(f'Loaded pretrained embeddings at {PRETRAINED_PATH}')
            return weights
        else:
            return None

    def init_vocab(self, dataset, weights=None):
        if weights is not None:
            tokens = [weights.index_to_key[i] for i in range(weights.vectors.shape[0])]
            return Vocabulary(tokens, min_freq=1, unk_token=UNK_WORD)
        else:
            return Vocabulary(dataset.tokens, min_freq=1, unk_token=UNK_WORD)

    def expert_postprocess(self, pred_lbl):
        for sentence in pred_lbl:
            for i, token in enumerate(sentence):
                if token.startswith("I-"):
                    if i == 0:
                        sentence[i] = "B-" + token[2:]
                    elif sentence[i-1] == "O":
                        sentence[i] = "B-" + token[2:]
                    elif (sentence[i-1].startswith("I-") or sentence[i-1].startswith("B-")) and sentence[i-1][2:] != sentence[i][2:]:
                        sentence[i] = "I-" + sentence[i-1][2:]  

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(self.device)))
        self.model.eval()
        sentences = []
        offsets = []
        for list in tokens:
            index_phrase = torch.tensor(Vocabulary.word_to_index(list, self.vocab), dtype=torch.int64)
            sentences.append(index_phrase)
            offsets.append(index_phrase.size(0))
        sentences = torch.cat(sentences)
        predicted_indexes = self.model(sentences)
        predicted_labels = [Vocabulary.index_to_tags(phrase.tolist(), self.tags) for phrase in torch.split(predicted_indexes.argmax(1), offsets)]
        self.expert_postprocess(predicted_labels)
        return predicted_labels
