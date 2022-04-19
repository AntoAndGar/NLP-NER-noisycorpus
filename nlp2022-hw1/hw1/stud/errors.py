from typing import List
from data_loading import *
from ner_pretrain_noisycorpus import NERv1_PRE
from rich.progress import track
from vocabulary import Vocabulary
from seqeval.metrics import accuracy_score, f1_score
from gensim import KeyedVectors
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
        self.model = NERv1_PRE(len(self.vocab), self.emb_size, self.hidden_size, self.num_layers, self.num_classes, self.bidirectional, device, self.weights)
        print("Model loaded")

    def init_weights(self, pretrained = False):
        if pretrained:
            weights = KeyedVectors.load(PRETRAINED_PATH)
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
        predicted_labels = [Vocabulary.index_to_tags(phrase.tolist(), self.tags) for phrase in torch.split(predicted_indexes.argmax(1), offsets)]
        self.expert_postprocess(predicted_labels)
        return predicted_labels

def extract_errors():
    valid_dataset = DatasetNER(DEV_PATH, verbose=True) 

    labels_s = valid_dataset.labels
    tokens_s = valid_dataset.tokens

    predictions_s = []
    batch_size = 32

    model = StudentModel("cpu")
    for i in track(range(0, len(tokens_s), batch_size), description="Evaluating errors..."):
        batch = tokens_s[i : min(i + batch_size, len(tokens_s))]
        batch_l = labels_s[i : min(i + batch_size, len(labels_s))]
        predictions_s = model.predict(batch)
        #accumulate predictions and labels with the minimum f1 score
        for j in range(len(batch)):
            f1 = f1_score([batch_l[j]], [predictions_s[j]], average="macro")
            if f1 != 1:
                # write the predictions, f1 score and labels to the file
                with open("../errors.txt", "a", encoding="utf-8") as f:
                    f.write(f"{f1}\n{batch[j]}\n{Vocabulary.index_to_words(Vocabulary.word_to_index(batch[j], model.vocab), model.vocab)}\npredicted:   {predictions_s[j]}\nground truth:{batch_l[j]}\n\n")
                f.close()
    print("Done!")

extract_errors()