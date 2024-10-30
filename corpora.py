import re
from typing import Generator, Any, Literal

from nltk.corpus import brown  

class Corpus:
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, sentence: list[str]) -> str:
        raise NotImplementedError

    def __iter__(self) -> Generator[str, Any, None]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError      

class BrownDataset(Corpus):
    def __init__(self) -> None:
        super().__init__()
        self.sentences = brown.sents()

    def preprocess(self, sentence: list[str]) -> str:
        result = ' '.join(sentence)
        result = re.sub(r'\s([.,!?;:])', r'\1', result)
        result = re.sub('`` ', '"', result)
        result = re.sub(" ''", '"', result)
        result = re.sub(r'(?<!\.)\.\.(?!\.)', r'.', result)
        result = re.sub(r'\( ', '(', result)
        result = re.sub(r' \)', ')', result)
        # result = re.sub(':', '', result) # must remove colons to graph
        return result
            
    def __iter__(self) -> Generator[str, Any, None]:
        for sentence in self.sentences:
            yield self.preprocess(sentence)
            
    def __str__(self) -> Literal['Brown_corpus']:
        return 'Brown_corpus'
