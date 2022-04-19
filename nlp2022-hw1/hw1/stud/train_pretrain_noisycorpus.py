import torch
import torch.nn as nn

from stud.ner_pretrain_noisycorpus import NERv1_PRE
from vocabulary import Vocabulary
from data_loading import *
from tqdm import tqdm
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from functools import partial
from torch.utils.data import DataLoader

from gensim.models import KeyedVectors
import numpy as np

from configuration import *

# reproducibility stuff
torch.manual_seed(42)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True  # This Deterministic mode can have a performance impact
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

#debuggability stuff
#torch.autograd.detect_anomaly()
#torch.set_anomaly_enabled(True)
#torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step(model, loss_function, label, sentence, offsets, f_acc, tot_len, optim=None):
    """
    One step of training

    :param model: model to train
    :param loss_function: loss function
    :param label: label of the sentence
    :param sentence: sentence to train on
    :param offsets: offsets of the sentence
    :param f_acc: f1-score accumulator
    :param tot_len: length of the sentence
    :param optim: optimizer

    :return: loss, f1-score, f1-score accumulator, length of the sentence
    """
    predicted_label = model(sentence)
    loss = loss_function(predicted_label, label)
    if optim is not None:
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optim.step()
        #optim.zero_grad()

    str_predicted_labels = [Vocabulary.index_to_tags(sentence.tolist(), TAG_DICT) for sentence in torch.split(predicted_label.argmax(1), offsets)]
    str_labels = [Vocabulary.index_to_tags(sentence.tolist(), TAG_DICT) for sentence in torch.split(label, offsets)]
    f1 = f1_score(str_labels, str_predicted_labels, average="macro")

    f_acc += f1
    tot_len += 1
    return loss.item(), f1, f_acc, tot_len

def train(model, loss_function, optim, train_dataloader, epoch):
    """
    Training function

    :param model: model to train
    :param loss_function: loss function
    :param optim: optimizer
    :param train_dataloader: training data
    :param epoch: number of current epoch

    :return: f1-score average
    """
    model.train()
    progress_bar_train = tqdm()
    f_acc_tr, tot_len_tr = 0, 0
    for (label, sentence, offsets) in train_dataloader:
        loss_tr, f1_tr, f_acc_tr, tot_len_tr = step(model, loss_function, label, sentence, offsets, f_acc_tr, tot_len_tr, optim)
        progress_bar_train.set_description(f'Epoch: {epoch} Loss:       {loss_tr:.4f} | F1-score:       {f1_tr:.4f}    ')       
        progress_bar_train.update()
        f_avg_train = f_acc_tr/tot_len_tr
    progress_bar_train.close()
    return f_avg_train

def evaluate(model, loss_function, valid_dataloader, epoch):
    """
    Evaluation function
    
    :param model: model to evaluate
    :param loss_function: loss function
    :param valid_dataloader: validation data
    :param epoch: number of current epoch

    :return: f1-score average
    """
    model.eval()
    progress_bar_val = tqdm()
    f_acc_val, tot_len_val = 0, 0
    with torch.no_grad():
        for (label, sentence, offsets) in valid_dataloader:
            loss_val, f1_score_val, f_acc_val, tot_len_val = step(model, loss_function, label, sentence, offsets, f_acc_val, tot_len_val)
            progress_bar_val.set_description(f'Epoch: {epoch} Loss Valid: {loss_val:.4f} | F1-score Valid: {f1_score_val:.4f}    ')
            progress_bar_val.update()
        f_avg_val = f_acc_val/tot_len_val
    progress_bar_val.close()
    return f_avg_val


def fit(epochs, model, loss_function, optim, sched, train_dataloader, valid_dataloader):
    """
    Training loop
    
    :param epochs: number of epochs
    :param model: model to train
    :param loss_function: loss function
    :param optim: optimizer
    :param sched: scheduler
    :param train_dataloader: dataloader for training
    :param valid_dataloader: dataloader for validation

    :return: model, loss, f1-score, f1-score accumulator, length of the sentence
    """
    best_f1_val = 0
    for epoch in range(1, epochs + 1):
        # training phase
        f1_avg_train = train(model, loss_function, optim, train_dataloader, epoch)

        # validation phase
        f1_avg_val = evaluate(model, loss_function, valid_dataloader, epoch)

        print(f'Epoch: {epoch} F1-avg Train: {f1_avg_train:.4f} | F1-avg Valid: {f1_avg_val:.4f}')
        sched.step(f1_avg_val)
        if f1_avg_val > best_f1_val:
            best_f1_val = f1_avg_val
            print(f'Saved best model at epoch: {epoch}')
            torch.save(model.state_dict(), SAVING_PATH + f'model_best.pth')
            print('Saved model')
    return best_f1_val

def main():
    # load data and create train and dev datasets
    train_dataset = DatasetNER(TRAINING_PATH, verbose=True) 
    valid_dataset = DatasetNER(DEV_PATH, verbose=True) 

    ########
    #code from NLP Notebook #8 
    #weights = gensim.downloader.load("glove-wiki-gigaword-300")
    #vectors = weights.vectors
    #print(vectors.shape)
    # mean vector for unknowns
    #unk = np.mean(vectors, axis=0, keepdims=True)
    ######## 
    #weights.add_vectors(UNK_WORD, unk)
    #weights.save(PRETRAINED_PATH)
    #print(f'Saved pretrained embeddings at {PRETRAINED_PATH}')

    # load pretrained embeddings
    weights = KeyedVectors.load(PRETRAINED_PATH)
    print(f'Loaded pretrained embeddings at {PRETRAINED_PATH}')

    tokens = [weights.index_to_key[i] for i in range(weights.vectors.shape[0])]
    # create vocabulary
    vocab = Vocabulary(tokens, min_freq=1, unk_token=UNK_WORD)
    assert len(tokens) == len(vocab.vocab)

    print("vocabulary len:", len(vocab.vocab))
    num_classes = len(TAG_DICT)

    def collate_batch(batch, vocab):
        """ 
        Create tensors for batching data:
        1st is a tensor containg all the label for each word
        2nd is a tensor containg all the words of batch number of sentences
        3rd is a tensor of offsets indexing from start to finish the index of a sentence
        """
        label_list, sentence_list, offsets = [], [], []
        for (tokenized_sentence, labels) in batch:
            label_list.append(torch.tensor(Vocabulary.tag_to_index(labels, TAG_DICT), dtype=torch.int64))
            processed_sentence = torch.tensor(Vocabulary.word_to_index(tokenized_sentence, vocab), dtype=torch.int64)
            sentence_list.append(processed_sentence)
            offsets.append(processed_sentence.size(0))
        label_list = torch.cat(label_list)
        sentence_list = torch.cat(sentence_list)
        return label_list.to(device), sentence_list.to(device), offsets#.to(device)

    # instantiate model
    model = NERv1_PRE(len(vocab), EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, BIDIRECTIONAL, device, weights).to(device)
    #model = GRU(len(vocab), EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, device).to(device)

    # instantiate loss function
    if WEIGHTS is not None:
        loss_function = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHTS).to(device)).to(device)
    else:
        #loss_function = nn.NLLLoss().to(device)
        loss_function = nn.CrossEntropyLoss().to(device)
    
    # instantiate optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    #optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay=WEIGHT_DECAY)

    # instantiate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.6, patience=2, cooldown=1, min_lr=0.00001,  threshold=0.00002, verbose=True)

    # create dataloaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=partial(collate_batch, vocab=vocab))
                                #num_workers=4, persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=partial(collate_batch, vocab=vocab))
                                #num_workers=4, persistent_workers=True)

    # train model
    best_f1 = fit(EPOCHS, model, loss_function, optimizer, scheduler, train_dataloader, valid_dataloader)
    print("model with best f1:", best_f1)

    # save model
    torch.save(model.state_dict(), SAVING_PATH+'model.pth')
    print('Saved last model')

if __name__ == '__main__':
    main()