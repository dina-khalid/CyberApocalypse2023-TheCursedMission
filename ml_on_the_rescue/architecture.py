	import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

vocab = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&'()*+,-./:;<=>?@[\]^_`{|}\n")

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx) 
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits
  
    def generate_text(self, text, length):
        # convert text to tensor
        text = torch.tensor([vocab.index(c) for c in text], dtype=torch.long)
        # # generate the text
        for i in range(length):
            # get the prediction
            prediction = model(text.unsqueeze(0), None)
            # # get the prediction of the last two characters
            prediction = prediction[0, -1]
            # # get the index of the highest probability
            index = torch.argmax(prediction)
            # # get the character from the vocab
            char = vocab[index]
            # # append the character to the text
            text = torch.cat((text, torch.tensor([index], dtype=torch.long)))
        # # decode the output
        return "".join([vocab[i] for i in text])
# decode the output
model = BigramLanguageModel(len(vocab))
model.load_state_dict(torch.load("bigram_model.pt"))
print(model.eval())
print(model.generate_text(vocab, len(vocab)))
