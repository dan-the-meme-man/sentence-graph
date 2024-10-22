import os

from tokenization import *
from corpora import *
from graph import *

bert_tokenizer = Tokenizer('bert-base-uncased')
# gpt2_tokenizer = Tokenizer('gpt2')

corpus = BrownDataset()

def load_or_make_graph(tokenizer, corpus):
    os.makedirs('graphs', exist_ok=True)
    if not os.path.exists(os.path.join('graphs', f'{tokenizer}_{corpus}.graphml')):
        return LMGraph(tokenizer, corpus)
    else:
        return LMGraph(
            tokenizer,
            corpus,
            G=nx.read_graphml(os.path.join('graphs', f'{tokenizer}_{corpus}.graphml'))
        )

bert_brown_graph = load_or_make_graph(bert_tokenizer, corpus)

bert_brown_graph.draw()