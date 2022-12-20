import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm 
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv2(x, edge_index)
        return x
        # return F.log_softmax(x, dim=1)


def GCNTraining(data, verbose = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (verbose):
        print('Training GCN on ',device)
    model = GCN(num_node_features = data.num_node_features, hidden_channels = 16, num_classes = data.num_classes).to(device)
    data = data[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    total_loss = 0
    for epoch in tqdm(range(100)):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        if (verbose):
            if epoch%5 == 0:
                print('epoch: ',epoch,' Loss = ',total_loss.item())
    return total_loss, out
