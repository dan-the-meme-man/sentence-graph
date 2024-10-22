from transformers import AutoTokenizer

class Tokenizer:
    
    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.vocab = self.tokenizer.get_vocab()
        if self.tokenizer.bos_token is None:
            try:
                self.bos_token = self.tokenizer.cls_token
            except:
                raise ValueError('Tokenizer does not have a BOS token.')
        else:
            self.bos_token = self.tokenizer.bos_token
        
    def __str__(self):
        return self.tokenizer.__class__.__name__
    
    def tokenize(self, sentence):
        return [self.bos_token] + self.tokenizer.tokenize(sentence)