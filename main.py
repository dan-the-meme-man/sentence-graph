import os
import networkx as nx

from graph import LMGraph
from tokenization import Tokenizer
from corpora import Corpus, BrownDataset
from graph import LMGraph

bert_tokenizer = Tokenizer('bert-base-uncased')
# gpt2_tokenizer = Tokenizer('gpt2')

corpus = BrownDataset()

def load_or_make_graph(tokenizer: Tokenizer, corpus: Corpus) -> LMGraph:
    os.makedirs('graphs', exist_ok=True)
    if not os.path.exists(os.path.join('graphs', f'{tokenizer}_{corpus}.graphml')):
        the_graph = LMGraph(tokenizer, corpus)
        nx.write_graphml(the_graph.G, os.path.join('graphs', f'{tokenizer}_{corpus}.graphml'))
        return the_graph
    else:
        return LMGraph(
            tokenizer,
            corpus,
            G=nx.read_graphml(os.path.join('graphs', f'{tokenizer}_{corpus}.graphml')),
            from_file=True
        )

bert_brown_graph = load_or_make_graph(bert_tokenizer, corpus)

bert_brown_graph.draw()
