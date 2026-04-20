import numpy as np
import pandas as pd
import torch
import pickle
from transformers import BertTokenizerFast, AutoModel
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW 
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def main():
    # Import cleaned dataset
    try:
        cleaned_dataset = pd.read_csv('../data/staged/cleaned_ratings_and_reviews.csv')
    except:
        cleaned_dataset = pd.read_csv('data/staged/cleaned_ratings_and_reviews.csv')
    
    # For some reason the fillna doesn't transfer over from data_validation
    cleaned_dataset['comments'] = cleaned_dataset['comments'].fillna('').astype(str)
    
    mapping = {"Under": 0, "Top": 1}
    # Convert strings to ints
    cleaned_dataset['performance'] = [mapping[label] for label in cleaned_dataset['performance']]
    
    # Separate Text into Training (70%), Validation (Hyperparmeter Tuning) (15%), and Testing (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(cleaned_dataset['comments'], cleaned_dataset['performance'],
                                                        random_state = 28,
                                                        test_size = 0.3,
                                                        stratify = cleaned_dataset['performance'])
                                                        
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    random_state = 28,
                                                    test_size = 0.5,
                                                    stratify = y_temp)
    
    rus = RandomUnderSampler(sampling_strategy='majority')
    X_train, y_train = rus.fit_resample(X_train.to_frame(), y_train.to_frame())
    X_train = X_train['comments']
    y_train = y_train['performance']
    
    print(f"Resampled shape: {y_train.value_counts()}")
    
    # Load pre-trained BERT model and tokenizer
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Padding Length (Hyperparameter): Average text length was calculated to be 51 after plotting in separate program.
    pad_len = 51
    
    # Tokenize the data
    tokens_train = tokenizer(
        X_train.tolist(),
        max_length = pad_len,
        padding = True,
        truncation = True
    )

    tokens_val = tokenizer(
        X_val.tolist(),
        max_length = pad_len,
        padding = True,
        truncation = True
    )

    tokens_test = tokenizer(
        X_test.tolist(),
        max_length = pad_len,
        padding = True,
        truncation = True
    )

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(y_train.tolist())
    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_dataloader = DataLoader(train_data,
                                  shuffle = True,
                                  batch_size = 16,
                                  drop_last = True)
    
    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(y_val.tolist())
    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_dataloader = DataLoader(val_data,
                                shuffle = True,
                                batch_size = 16,
                                drop_last = True)
    
    # Save for testing
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(y_test.tolist())
    try:
        torch.save({'test_seq': test_seq, 'test_mask': test_mask, 'test_y': test_y}, '../data/artifacts/test_data.pt')
    except:
        torch.save({'test_seq': test_seq, 'test_mask': test_mask, 'test_y': test_y}, 'data/artifacts/test_data.pt')
    
    # Freeze the pretrained layers
    for param in bert.parameters():
        param.requires_grad = False
    # Now instantiate the model
    model = BERT_architecture(bert)
    # Instaniate an optmizier to optmize computing the loss function
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    # Train the Model in a Training Loop (Iterate through dataset, calculate loss, update weights, repeat)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Maximum number of epochs
    max_batches = 500
    
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        # clear previously calculated gradients
        model.zero_grad()
        # get model predictions for the current batch
        preds = model(sent_id, mask)
        # compute the loss between actual and predicted values
        loss = criterion(preds, labels)
        # backward pass to calculate the gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        if step >= max_batches:
            break
        
    # Finally, save model for evaluation
    try:
        with open("../data/artifacts/bert_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("BERT Model Successfully Saved")
    except:
        with open("data/artifacts/bert_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("BERT Model Successfully Saved")
    
#defining new layers
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
