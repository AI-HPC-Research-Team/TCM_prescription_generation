
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch.utils.data as Data
from sklearn.metrics import r2_score

node_embedding = torch.load('node_embedding.pt').tolist()

def generate_dataset(dataset, graph_embedding):
    data = []
    target = []
    Node = pd.read_csv('Nodes.csv', encoding='gbk')
    nodes = Node['Nodes'].tolist()
    node_index = {}
    for i in range(len(nodes)):
        node_index.update({nodes[i]: i})
    for index, row in dataset.iterrows():
        compounds = row['Compounds'].split(',')
        target.append([float(row['Score'])])
        prescription = []
        for i in compounds:
            prescription.append(graph_embedding[node_index[i]])
        pres = torch.mean(torch.tensor(prescription), dim=0)
        data.append(pres.tolist())
    return torch.tensor(data), torch.FloatTensor(target)

trainset = pd.read_csv('trainset.csv', encoding='gbk')
validateset = pd.read_csv('valset.csv', encoding='gbk')
testset = pd.read_csv('testset.csv', encoding='gbk')
train_data, train_target = generate_dataset(trainset,node_embedding)
validate_data, validate_target = generate_dataset(validateset,node_embedding)
test_data, test_target = generate_dataset(testset,node_embedding)
traindata = Data.TensorDataset(train_data, train_target)
validatedata = Data.TensorDataset(validate_data, validate_target)
testdata = Data.TensorDataset(test_data, test_target)
train_loader = Data.DataLoader(
    dataset=traindata,
    batch_size=128,
    shuffle=True,
)
val_loader = Data.DataLoader(
    dataset=validatedata,
    batch_size=128,
    shuffle=True,
)
test_loader = Data.DataLoader(
    dataset=testdata,
    batch_size=128,
    shuffle=False,
)

class Recommender(nn.Module):
    def __init__(self, in_features, hidden_features1=None, hidden_features2=None,hidden_features3=None,
                 hidden_features4=None,hidden_features5=None, out_features=None):
        super().__init__()
        self.hidden1 = nn.Linear(in_features,hidden_features1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_features1, affine=True)
        self.hidden2 = nn.Linear(hidden_features1, hidden_features2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_features2, affine=True)
        self.hidden3 = nn.Linear(hidden_features2, hidden_features3)
        self.batchnorm3 = nn.BatchNorm1d(hidden_features3, affine=True)
        self.hidden4 = nn.Linear(hidden_features3, hidden_features4)
        self.batchnorm4 = nn.BatchNorm1d(hidden_features4, affine=True)
        self.hidden5 = nn.Linear(hidden_features4, hidden_features5)
        self.batchnorm5 = nn.BatchNorm1d(hidden_features5, affine=True)
        self.batchnorm6 = nn.BatchNorm1d(out_features, affine=True)
        self.out = nn.Linear(hidden_features5,out_features)

    def forward(self, X, drop=0.):
        X = self.hidden1(X)
        X = self.batchnorm1(X)
        X = self.hidden2(F.gelu(X))
        X = F.dropout(X,p=drop)
        X = self.batchnorm2(X)
        X = self.hidden3(F.gelu(X))
        X = F.dropout(X,p=drop)
        X = self.batchnorm3(X)
        X = self.hidden4(F.gelu(X))
        X = F.dropout(X,p=drop)
        X = self.batchnorm4(X)
        X = self.hidden5(F.gelu(X))
        X = F.dropout(X,p=drop)
        X = self.batchnorm5(X)
        X = self.out(F.gelu(X))
        X = self.batchnorm6(X)
        X = F.sigmoid(X)
        return X

batch_size, lr, num_epochs = 256, 0.01, 1000
net = Recommender()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=lr)


def train(epoch):
    net.train()
    running_loss = 0.0
    targets_train = []
    predict_train = []
    for inputs,target in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        targets_train.extend(target.tolist())
        predict_train.extend(outputs.tolist())
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    trainacc = r2_score(torch.FloatTensor(targets_train).detach().numpy(), torch.FloatTensor(predict_train).detach().numpy())
    print('Accuracy on train set: ',trainacc)
    print('[%d]loss: %.3f' %(epoch+1,running_loss/len(train_loader)))

def test():
    predict_test = []
    target_test = []
    with torch.no_grad():
        for input,tar in test_loader:
            out = net(input)
            target_test.extend(tar.tolist())
            predict_test.extend(out.tolist())
        testacc = r2_score(torch.FloatTensor(target_test).detach().numpy(), torch.FloatTensor(predict_test).detach().numpy())
    print('Accuracy on test set: ', testacc)

val_max = np.inf
for epoch in range(num_epochs):
    train(epoch)
    val_loss = 0
    val_acc = 0
    net.eval()
    targets = []
    predict = []
    for inputs,target in val_loader:
        outputs = net(inputs)
        vloss = criterion(outputs, target)
        val_loss += vloss.item()
        targets.extend(target.tolist())
        predict.extend(outputs.tolist())
    print('[%d]vloss: %.3f' %(epoch+1,val_loss/len(val_loader)))
    test()
    if val_loss/len(val_loader) < val_max:
        tmp = val_max
        val_max = val_loss / len(val_loader)
        print('vloss down from',str(tmp),'--------->',str(val_max))
        torch.save(net.state_dict(), 'recommender_parameter.pkl')

