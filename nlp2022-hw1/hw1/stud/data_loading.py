import torch

from typing import Tuple, List

class DatasetNER(torch.utils.data.Dataset):
    '''
    Creates a dataset object storing a list of tokens and a list of labels.

    Args:
        path: path to the dataset file
        verbose: if True, prints informations about the dataset
    '''
    
    def __init__(self, path, verbose = False):
        super().__init__()
        self.verbose = verbose
        self.tokens, self.labels = self.read_dataset(path)

    def __getitem__(self, index):
        return (self.tokens[index], self.labels[index])
    
    def __len__(self):
        return len(self.tokens)

    #########
    # method copied from evaluate, with some fixes in the encoding reading part, all credit to TAs
    def read_dataset(self, path: str) -> Tuple[List[List[str]], List[List[str]]]:
        tokens_s = []
        labels_s = []

        tokens = []
        labels = []

        # fixed utf-8 encoding for some letters
        with open(path, mode="r", encoding="utf-8") as f:
            for line in f:

                line = line.strip()
                
                if line.startswith("#\t"):
                    tokens = []
                    labels = []
                elif line == "":
                    tokens_s.append(tokens)
                    labels_s.append(labels)
                else:
                    token, label = line.split("\t")
                    tokens.append(token)
                    labels.append(label)

        assert len(tokens_s) == len(labels_s)

        if self.verbose:
            print(f'len sentences: {len(tokens_s)}, len label: {len(labels_s)}')
        
        return tokens_s, labels_s
