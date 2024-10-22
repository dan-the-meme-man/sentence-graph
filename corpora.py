import re
from nltk.corpus import brown

class Dataset:
    def __init__(self):
        pass
    
    def preprocess(self, sentences):
        pass
    
    def __iter__(self):
        for sentence in self.sentences:
            yield sentence
        

class BrownDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.sentences = self.preprocess(brown.sents())

    def preprocess(self, sentences):
        for sentence in sentences:
            result = ' '.join(sentence)
            result = re.sub(r'\s([.,!?;:])', r'\1', result)
            result = re.sub('`` ', '"', result)
            result = re.sub(" ''", '"', result)
            result = re.sub(r'(?<!\.)\.\.(?!\.)', r'.', result)
            result = re.sub(r'\( ', '(', result)
            result = re.sub(r' \)', ')', result)
            yield result
            
    def __str__(self):
        return 'Brown corpus'
