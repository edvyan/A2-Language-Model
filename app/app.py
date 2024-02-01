import torch
import torch.nn as nn
import math
from dash import dcc, html, callback, Input, Output
import dash
import torchtext, math


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim    = hid_dim
        self.emb_dim    = emb_dim
        
        self.embedding  = nn.Embedding(vocab_size, emb_dim)
        self.lstm       = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hid_dim, vocab_size) # fc is the last layer for 
        
        self.init_weights()
    
    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim,
                self.hid_dim).uniform_(-init_range_other, init_range_other) #W_e
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim,   
                self.hid_dim).uniform_(-init_range_other, init_range_other) #W_h
    
    def init_hidden(self, batch_size, device): 
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
        
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach() #not to be used for gradient computation
        cell   = cell.detach()
        return hidden, cell
        
    def forward(self, src, hidden):
        #src: [batch_size, seq len]
        embedding = self.dropout(self.embedding(src)) #harry potter is
        #embedding: [batch-size, seq len, emb dim]
        output, hidden = self.lstm(embedding, hidden)
        #ouput: [batch size, seq len, hid dim]
        #hidden: [num_layers * direction, seq len, hid_dim]
        output = self.dropout(output)
        prediction =self.fc(output)
        #prediction: [batch_size, seq_len, vocab_size]
        return prediction, hidden

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens


# Load your model, tokenizer, and vocabulary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the vocabulary from the saved file
vocab = torch.load('vocab.pt')
vocab_size = len(vocab)
emb_dim = 1024                
hid_dim = 1024               
num_layers = 2           
dropout_rate = 0.65          
lr = 1e-3
device = 'cpu'

model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
model.load_state_dict(torch.load('best-val-lstm_lm.pt', map_location=device))
model.to(device)
model.eval()
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')    


app = dash.Dash(__name__)


app.layout = html.Div([
    html.Label('Enter a Text Prompt:'),
    dcc.Input(id='input-prompt', type='text', value=''),
    html.Div(id='output')
])


@callback(
    Output('output', 'children'),
    [Input('input-prompt', 'value')]
)

def generate_continuation(input_prompt):
    if not input_prompt:
        return ''
    
    max_seq_len = 50  # Set the maximum sequence length
    temperature = 1.0  # Set the temperature for generation

    # Generate text
    continuation = generate(input_prompt, max_seq_len, temperature, model, tokenizer, vocab, device)

    # Format the output
    continuation_text = ' '.join(continuation)

    return continuation_text


if __name__ == '__main__':
    app.run_server(debug=True)
