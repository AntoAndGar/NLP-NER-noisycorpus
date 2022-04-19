from collections import Counter
from logging.config import valid_ident
import matplotlib.pyplot as plt
from data_loading import *

from configuration import *

def count_label(label):
    # for each label in the set, count the number of times it appears
    data_counter = Counter()
    for elem in label:
        data_counter.update(elem)
    print(data_counter.most_common(100))
    data_counter.pop('O')
    return data_counter


def plot_bar(dict):
    # plot a distribution of data in the set
    plt.figure(figsize=(10,5))
    plt.bar(dict.keys(), dict.values())
    plt.legend()
    plt.show()

train_dataset = DatasetNER(TRAINING_PATH, verbose=True) 
valid_dataset = DatasetNER(DEV_PATH, verbose=True) 

l_train = count_label(train_dataset.labels)
plot_bar(l_train)

l_test = count_label(valid_dataset.labels)
plot_bar(l_test)

def plot_pie(dict):
    # plot a distribution of data in the training set
    plt.figure(figsize=(10,5))
    plt.pie(dict.values(), labels=dict.keys(), autopct='%1.1f%%')
    plt.legend()
    plt.show()

plot_pie(l_train)
plot_pie(l_test)
