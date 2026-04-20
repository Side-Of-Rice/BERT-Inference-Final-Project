import pandas as pd
import numpy as np
import pickle
import torch
import warnings
from sklearn.metrics import classification_report
from torch import nn

def main():
    warnings.filterwarnings("ignore")
    
    #Retrieve test dataset
    try:
        test_dict = torch.load('../data/artifacts/test_data.pt')
    
        #Retrieve trained BERT model
        with open('../data/artifacts/bert_model.pkl', 'rb') as file:
            bert = pickle.load(file)
    except:
        test_dict = torch.load('data/artifacts/test_data.pt')
    
        #Retrieve trained BERT model
        with open('data/artifacts/bert_model.pkl', 'rb') as file:
            bert = pickle.load(file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_size = 1000

    # Move the model to the selected device (GPU or CPU)
    bert.to(device)

    # Get predictions for test data
    with torch.no_grad():
        # Move data to device and perform prediction
        preds = bert(test_dict['test_seq'][:test_size].to(device), test_dict['test_mask'][:test_size].to(device))
    
        # Move the predictions to CPU for further processing
        preds = preds.detach().cpu().numpy()

    # Convert predictions to class labels (0 or 1 for binary classification)
    pred = np.argmax(preds, axis=1)

    # Print classification report
    print(classification_report(test_dict['test_y'][:test_size], pred))
    print(pred)
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

if __name__ == "__main__":
    main()
