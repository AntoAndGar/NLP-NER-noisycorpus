# Adapted from a personal database of code, frankly I don't remember the reference
from sklearn.metrics import confusion_matrix
from data_loading import *

import numpy as np
import matplotlib.pyplot as plt
import itertools
from rich.progress import track
from  vocabulary import Vocabulary
from hw1.stud.ner_pretrain_stdbatch import NERv2
from gensim.models import KeyedVectors
import torch

from typing import List, Any, Dict
from configuration import *

class StudentModel():

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
        self.model = NERv2(len(self.vocab), self.emb_size, self.hidden_size, self.num_layers, self.num_classes, self.bidirectional, device, self.weights)
        print("Model loaded")

    def init_weights(self, pretrained = False):
        if pretrained:
            weights = KeyedVectors.load(PRETRAINED_PATH)
            pad = np.zeros_like(weights.get_vector("the"))
            weights.add_vectors(PAD_TOKEN, pad)
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
                        print("I- tag at the beginning of sentence")
                    elif sentence[i-1] == "O":
                        sentence[i] = "B-" + token[2:]
                        print("I- tag without B- tag")
                    elif (sentence[i-1].startswith("I-") or sentence[i-1].startswith("B-")) and sentence[i-1][2:] != sentence[i][2:]:
                        sentence[i] = "I-" + sentence[i-1][2:]
                        print("I- tag with different tag")

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
        predicted_labels = [Vocabulary.index_to_tags(phrase.tolist(), self.tags) for phrase in torch.split(predicted_indexes.argmax(-1), offsets)]
        self.expert_postprocess(predicted_labels)
        return predicted_labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm = np.around(cm, decimals=3)
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(spacing='proportional')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    frmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], frmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def flat_list(l: List[List[Any]]) -> List[Any]:
    return [_e for e in l for _e in e]


def count(l: List[Any]) -> Dict[Any, int]:
    d = {}
    for e in l:
        d[e] = 1 + d.get(e, 0)
    return d

def confusion_matrix_plot(normalize = False):
    # plot a confusion matrix of the result of the tags in the validation set

    valid_dataset = DatasetNER(DEV_PATH, verbose=True) 

    labels_s = valid_dataset.labels
    tokens_s = valid_dataset.tokens

    predictions_s = []
    batch_size = 32

    
    model = StudentModel("cpu")
    for i in track(range(0, len(tokens_s), batch_size), description="Confusion matrix"):
        batch = tokens_s[i : i + batch_size]
        predictions_s += model.predict(batch)

    flat_labels_s = flat_list(labels_s)
    flat_predictions_s = flat_list(predictions_s)

    label_distribution = count(flat_labels_s)
    pred_distribution = count(flat_predictions_s)

    print(f"# instances: {len(flat_list(labels_s))}")

    keys = set(label_distribution.keys()) | set(pred_distribution.keys())
    for k in keys:
        print(
            f"\t# {k}: ({label_distribution.get(k, 0)}, {pred_distribution.get(k, 0)})"
        )
    
    cm = confusion_matrix(flat_labels_s, flat_predictions_s, labels=["B-CORP", "B-CW", "B-GRP", "B-LOC", "B-PER", "B-PROD", "I-CORP", "I-CW", "I-GRP", "I-LOC", "I-PER", "I-PROD", "O"])
    plot_confusion_matrix(cm, classes=["B-CORP", "B-CW", "B-GRP", "B-LOC", "B-PER", "B-PROD", "I-CORP", "I-CW", "I-GRP", "I-LOC", "I-PER", "I-PROD", "O"], normalize=normalize, title='Confusion matrix')
    plt.show()

confusion_matrix_plot(True)