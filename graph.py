import os
from copy import deepcopy

import networkx as nx
from networkx import DiGraph
from networkx.drawing.nx_pydot import pydot_layout
import matplotlib.pyplot as plt

from tokenization import Tokenizer
from corpora import Corpus

class LMGraph:

    def __init__(
        self,
        tokenizer: Tokenizer,
        corpus: Corpus,
        G: nx.DiGraph = None,
        from_file: bool = False
    ) -> None:
        
        if from_file:
            print(f'Loading graph from file {tokenizer}_{corpus}.graphml.')
        else:
            print(f'Building graph with {tokenizer} tokenizer and {corpus}.')
        
        self.tokenizer_name = str(tokenizer)
        self.corpus_name = str(corpus)
        self.tokenizer = tokenizer
        
        if G is not None:
            self.G = G
        else:
            self.G = nx.DiGraph()

            for word in tokenizer.vocab.keys():
                self.G.add_node(word)
            print(f'Added {len(self.G.nodes)} nodes.')
                
            for i, sent in enumerate(corpus):
                tokens = tokenizer.tokenize(sent)
                # print(tokens)
                # exit()
                for j in range(1, len(tokens) - 1):
                    if self.G.has_edge(tokens[j-1], tokens[j]):
                        self.G[tokens[j-1]][tokens[j]]['weight'] += 1
                    else:
                        # print(tokens[j-1], tokens[j])
                        self.G.add_edge(tokens[j-1], tokens[j], weight=1)
                if (i+1) % 1000 == 0:
                    print(f'Processed {i+1} sentences.')
            print(f'Added {len(self.G.edges)} edges.')
            isolates = list(nx.isolates(self.G))
            print(f'Removing {len(isolates)} isolates.')
            self.G.remove_nodes_from(isolates)
            print(f'Removing isolated self-loops...')
            self_loops = list(nx.selfloop_edges(self.G))
            for node in set(u for u, _ in self_loops):
                if self.G.in_degree(node) <= 1 and self.G.out_degree(node) <= 1:
                    self.G.remove_edge(node, node)
                    
        # self.check_node_types()



    def create_pruned_graph(self, max_edges: int) -> DiGraph:
        print(f'Pruning graph to {max_edges} edges.')
        sorted_edges = sorted(
            self.G.edges(data=True),
            key=lambda x: x[2]['weight'],
            reverse=True
        )
        G_pruned = deepcopy(self.G)
        for edge in sorted_edges[max_edges:]:
            G_pruned.remove_edge(edge[0], edge[1])
        isolates = list(nx.isolates(G_pruned))
        print(f'Removing {len(isolates)} isolates.')
        G_pruned.remove_nodes_from(isolates)
        self_loops = list(nx.selfloop_edges(G_pruned))
        for node in set(u for u, _ in self_loops):
            if G_pruned.in_degree(node) <= 1 and G_pruned.out_degree(node) <= 1:
                G_pruned.remove_edge(node, node)
        return G_pruned



    def check_node_types(self) -> None:
        for node in self.G.nodes:
            print(f'Node: {node}, Type: {type(node)}')



    def draw(
        self,
        with_weights: bool = False,
        with_labels: bool = False, 
        max_edges: int = 1_000
    ) -> None:
        G = self.create_pruned_graph(max_edges)
        print('Laying out graph...')
        # pos = nx.spring_layout(G, iterations=100)
        # pos = nx.kamada_kawai_layout(G)
        pos = pydot_layout(G, prog='dot', root=self.tokenizer.bos_token_id)
        # pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        print(f'Drawing graph with {len(G.nodes)} nodes and {len(G.edges)} edges...')
        nx.draw(
            G,
            pos,
            with_labels=with_labels,
            node_size=10,
            edge_color='gray',
            alpha=0.5
        )
        nx.draw(
            G.subgraph(self.tokenizer.bos_token_id),
            pos,
            node_color='red',
            node_size=15
        )
        if with_weights:
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        fig_path = os.path.join(
            'plots',
            f'{self.tokenizer_name}_{self.corpus_name}_{max_edges}.png'
        )
        print(f'Saving drawing to {fig_path}...')
        plt.title(f'{self.tokenizer_name} tokenizer on {self.corpus_name}')
        os.makedirs('plots', exist_ok=True)
        plt.savefig(fig_path)
