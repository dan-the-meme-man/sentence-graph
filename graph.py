import networkx as nx
import matplotlib.pyplot as plt

class LMGraph:
    
    def __init__(self, tokenizer, corpus, G=None):
        
        print(f'Building graph with {tokenizer} tokenizer and {corpus}.')
        
        self.tokenizer_name = str(tokenizer)
        self.corpus_name = str(corpus)
        
        if G is not None:
            self.G = G
        else:
            self.G = nx.DiGraph()

            for word in tokenizer.vocab.keys():
                self.G.add_node(word)
            print(f'Added {len(self.G.nodes)} nodes.')
                
            for i, sent in enumerate(corpus):
                tokens = [tokenizer.bos_token] + tokenizer.tokenize(sent)
                # print(tokens)
                for j in range(1, len(tokens) - 1):
                    if self.G.has_edge(tokens[j-1], tokens[j]):
                        self.G[tokens[j-1]][tokens[j]]['weight'] += 1
                    else:
                        # print(tokens[j-1], tokens[j])
                        self.G.add_edge(tokens[j-1], tokens[j], weight=1)
                if (i+1) % 1000 == 0:
                    print(f'Processed {i+1} sentences.')
            print(f'Added {len(self.G.edges)} edges.')
                    
    def draw(self, with_weights=False, with_labels=False):
        print('Laying out graph...')
        pos = nx.spring_layout(self.G, iterations=10)
        print('Drawing graph...')
        nx.draw(
            self.G,
            pos,
            with_labels=with_labels,
            node_size=10,
            edge_color='gray',
            alpha=0.5
        )
        if with_weights:
            edge_labels = nx.get_edge_attributes(self.G, 'weight')
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        print('Showing graph...')
        plt.title(f'{self.tokenizer_name} tokenizer on {self.corpus_name}')
        plt.show()
