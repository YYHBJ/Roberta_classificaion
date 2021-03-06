import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import transformers
from transformers import *
from sklearn import preprocessing

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large-mnli")
bert = RobertaModel.from_pretrained("roberta-large-mnli")

#Path
df = pd.read_csv('csv/train.csv')
val= pd.read_csv('csv/valid.csv')
test = pd.read_csv('csv/test.csv')

#定義欄位
# emo = df['emotion']
# sentence = df['text']


le = preprocessing.LabelEncoder()
le.fit(df['emotion'])
c = list(le.classes_)
labels={}
for idx, la in enumerate(c):
    labels.update({la:idx})
    
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, df):

        self.labels = [torch.tensor(labels[label], dtype=torch.long) for label in df['emotion']]
        self.texts  = [tokenizer(text, 
                               padding='max_length', max_length = 220, truncation=True,
                                return_tensors="pt") for text in df['text']]
        # self.story = torch.Tensor(tokenizer.convert_tokens_to_ids(self.texts)).long()

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class Classifier(nn.Module):
    
    def __init__(self, dropout=0.1):

        super(Classifier, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, 7)
        # self.linear2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        # linear_output2 = self.linear2(linear_output)
        final_layer = self.relu(linear_output)

        return final_layer

class Topic_Classifier(nn.Module):
    
    def __init__(self, dropout=0.1):

        super(Topic_Classifier, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, 128)
        # self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 10)
        # self.linear2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        # linear_output2 = self.linear2(linear_output)
        linear_output3 = self.linear3(linear_output)
        final_layer = self.relu(linear_output3)

        return final_layer
    
class Act_Classifier(nn.Module):
    
    def __init__(self, dropout=0.1):

        super(Act_Classifier, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(256, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        linear_output2 = self.linear2(linear_output)
        final_layer = self.relu(linear_output2)

        return final_layer
        
def train(model, train_data, val_data, learning_rate, epochs):
    
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=3, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    #GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')
    
    
                     
def evaluate(model, test_data):
    
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    all_testlabel=[]
    all_output=[]
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            
            #   print(test_label)
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
              
              all_testlabel.append(test_label.cpu())
              all_output.append(output.cpu().argmax(dim=1))
              
    import itertools
    from sklearn.metrics import f1_score, confusion_matrix
    testlabels=list(itertools.chain.from_iterable(all_testlabel))
    # print(testlabels)
    outputs = list(itertools.chain.from_iterable(all_output))
    f1_score = f1_score(testlabels, outputs,average='micro')
    confusion_matrix = confusion_matrix(testlabels, outputs)
    
              
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    print(f'f1 score:',f1_score)
    print(f'confusion_matrix:\n',confusion_matrix)
    
    
np.random.seed(2019)
print(len(df),len(val), len(test))


"""
TRAIN
"""
EPOCHS = 6
model = Classifier()
LR = 1e-6
              
train(model, df, val, LR, EPOCHS)
torch.save(model.state_dict(), 'models/model_emo_410.pt')
# torch.save(model,'models/model_emo_410.pt')  
evaluate(model, test)