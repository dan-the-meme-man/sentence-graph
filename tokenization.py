from transformers import AutoTokenizer

class Tokenizer:

    def __init__(self, name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.vocab = self.tokenizer.get_vocab()
        if self.tokenizer.bos_token is None:
            try:
                self.bos_token_id = self.tokenizer.cls_token_id
            except:
                self.bos_token_id = self.tokenizer.eos_token
        else:
            print('else')
            self.bos_token_id = self.tokenizer.bos_token_id
        print(f'BOS token: {self.tokenizer.bos_token}')
        
    def __str__(self) -> str:
        return self.tokenizer.__class__.__name__
    
    def tokenize(self, sentence: str) -> list[int | str | None]:
        return [self.bos_token_id] + self.tokenizer.encode(sentence, add_special_tokens=False)
