import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Loading BERT model.  Please be patient.")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

def load_checkpoint(load_path, model):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

model = BERT().to(device)

load_checkpoint('./BERT_output/model.pt', model)

inp = ""
print()
inp = input("Enter a string to be classified as a joke or not (type 'q' to quit): ")

while inp != "q":
    seq = torch.LongTensor([tokenizer.encode(inp, padding='max_length', max_length=128)])
    output = model(seq, torch.LongTensor([[1]]))
    _, output = output
    pred = torch.argmax(output, 1).tolist()[0]
    print("Joke" if pred == 1 else "Not Joke")
    inp = input("Enter a string to be classified as a joke or not (type 'q' to quit): ")
