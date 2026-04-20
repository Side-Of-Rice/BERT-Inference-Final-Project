from flask import Flask, request, jsonify
import pickle
import torch
import numpy as np
from torch import nn
from transformers import BertTokenizerFast

app = Flask(__name__)

# For pickel model
class BERT_architecture(nn.Module):

    def __init__(self, bert):

      super(BERT_architecture, self).__init__()

      self.bert = bert

      # dropout layer
      self.dropout = nn.Dropout(0.2)

      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)

      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)
      

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)

      # apply softmax activation
      x = self.softmax(x)
      
      return x

# Load a pre-trained model
try:
    with open('..\\data\\artifacts\\bert_model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    try:
        with open('data\\artifacts\\bert_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        #Docker Build (Linux)
        with open('data/artifacts/bert_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
@app.route('/predict', methods=['POST'])
def predict():
    new_review = list((request.data.decode('utf-8'),)) 
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    tokens_new = tokenizer(
        new_review,
        max_length = 200,
        padding = True,
        truncation = True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    new_seq = torch.tensor(tokens_new['input_ids'])
    new_mask = torch.tensor(tokens_new['attention_mask'])
    with torch.no_grad():
        prediction = model(new_seq.to(device), new_mask.to(device))
    
    prediction = prediction.detach().cpu().numpy()
    prediction = np.argmax(prediction, axis=1)
    
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    #Curl Example: curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: text/plain" -d "Your Text Here"
