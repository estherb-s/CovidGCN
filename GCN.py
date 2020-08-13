import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
# from torch_scatter import scatter_mean
from mol2graph import mol2vec
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from torchbearer import Trial

tasks = ['Active']  # Featurize dataset
dataset = pd.read_csv("data/trainingdata.csv")

msk = np.random.rand(len(dataset)) < 0.8
train_dataset = dataset[msk]
test_dataset = dataset[~msk]

train_mols = [Chem.MolFromSmiles(smiles) for smiles in train_dataset["SMILES"]]
test_mols = [Chem.MolFromSmiles(smiles) for smiles in test_dataset["SMILES"]]
print(train_mols)
quit()

# Convert molecule to graph w. the defined mol2vec fn & add label for training
train_X = [mol2vec(m) for m in train_mols]
# Assign Active label from train_dataset to the new train_X list
for i, data in enumerate(train_X):
  for index, row in train_dataset['Active'].iteritems():
      y = row
      data.y = torch.tensor([y], dtype=torch.long)
 
test_X = [mol2vec(m) for m in test_mols]
for i, data in enumerate(test_X):
  for index, row in test_dataset['Active'].iteritems():
      y = row
      data.y = torch.tensor([y], dtype=torch.long)
train_loader = DataLoader(train_X, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_X, batch_size=64, shuffle=True, drop_last=True)

print(train_X)
quit()
# ***Get features and labels from train_X and test_X ****

# Defined model architecture for GCN
n_features = 75
# definenet
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(n_features, 128, cached=False) # if you defined cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(128)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn2 = BatchNorm1d(64)
        self.fc1 = Linear(64, 64)
        self.bn3 = BatchNorm1d(64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 3)
         
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x 

# Train the model and evaluate the performance
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Train model for 200 epochs 
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Evaluate model's performance
model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))


# trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
# trial.with_generators(trainloader, test_generator=testloader)
# trial.run(epochs=10)
# results = trial.evaluate(data_key=torchbearer.TEST_DATA)
# print(results)

def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_X)
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)
hist = {"loss":[], "acc":[], "test_acc":[]}
for epoch in range(1, 101):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    hist["loss"].append(train_loss)
    hist["acc"].append(train_acc)
    hist["test_acc"].append(test_acc)
    print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Train_acc: {train_acc:.3}, Test_acc: {test_acc:.3}')
ax = plt.subplot(1,1,1)
ax.plot([e for e in range(1,101)], hist["loss"], label="train_loss")
ax.plot([e for e in range(1,101)], hist["acc"], label="train_acc")
ax.plot([e for e in range(1,101)], hist["test_acc"], label="test_acc")
plt.xlabel("epoch")
ax.legend()

# Average precision score and hit rate top 100
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
# Fit trained model
model.fit(train_dataset, nb_epoch=20)
y_true = np.squeeze(valid_dataset.y).astype(int)
print(y_true)
y_pred = model.predict(valid_dataset)[:,0,1].astype(int)
print(y_pred)
print("Average Precision Score:%s" % average_precision_score(y_true, y_pred))
sorted_results = sorted(zip(y_pred, y_true), reverse=True)
hit_rate_100 = sum(x[1] for x in sorted_results[:100]) / 100
print("Hit Rate Top 100: %s" % hit_rate_100)

# ROC-AUC score for training, test, validation datasets

y_prob = classifier.predict_proba(X_test)

macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")

metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification") #ADJUST****

print("Evaluating model")
# train_scores = model.evaluate(train_dataset, [metric], transformers)
# print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
# valid_scores = model.evaluate(valid_dataset, [metric], transformers)
# print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])

# test_scores = model.evaluate(test_dataset, [metric], transformers)
# print("Test ROC-AUC Score: %f" % test_scores["mean-roc_auc_score"])

# predicted_val = model.predict(valid_dataset)
# true_val = valid_dataset.y

# print(predicted_val.shape)
# print(true_val.shape)