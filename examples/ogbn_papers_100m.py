import os
import time
from typing import Optional
import argparse
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, GATConv

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--fan_out', type=int, default=16)
parser.add_argument(
    "--use_gat_conv",
    type=bool,
    default=False,
    help="Wether or not to use GATConv. (Defaults to using GCNConv)",
)
parser.add_argument(
    "--n_gat_conv_heads",
    type=int,
    default=4,
    help="If using GATConv, number of attention heads to use",
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-papers100M')
split_idx = dataset.get_idx_split()


def get_num_workers() -> int:
    try:
        return len(os.sched_getaffinity(0)) // 2
    except Exception:
        return os.cpu_count() // 2


kwargs = dict(
    data=dataset[0],
    num_neighbors=[50, 50],
    batch_size=128,
    num_workers=get_num_workers(),
)
train_loader = NeighborLoader(input_nodes=split_idx['train'], shuffle=True,
                              **kwargs)
val_loader = NeighborLoader(input_nodes=split_idx['valid'], **kwargs)
test_loader = NeighborLoader(input_nodes=split_idx['test'], **kwargs)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, use_gat_conv=False, n_gat_conv_heads=4):
        super().__init__()
        if use_gat_conv:
            self.conv1 = GATConv(in_channels, hidden_channels, heads=n_gat_conv_heads)
            self.conv2 = GATConv(hidden_channels, out_channels, heads=n_gat_conv_heads)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


model = GNN(dataset.num_features, 64, dataset.num_classes, args.use_gat_conv, args.n_gat_conv_heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

warmup_steps = 50

def train():
    model.train()

    for i, batch in enumerate(train_loader):
        if i >= warmup_steps:
            start_avg_time = time.perf_counter()
        start = time.perf_counter()
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y = batch.y[:batch.batch_size].view(-1).to(torch.long)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch: {epoch:02d}, Iteration: {i}, Loss: {loss:.4f}, '
                  f's/iter: {time.perf_counter() - start:.6f}')

    print(f'Average Training Iteration Time (s/iter): {time.perf_counter() - start_avg_time:.6f}')


@torch.no_grad()
def test(loader: NeighborLoader, eval_steps: Optional[int] = None):
    model.eval()

    total_correct = total_examples = 0
    for i, batch in enumerate(loader):
        if eval_steps is not None and i >= eval_steps:
            break
        if i >= warmup_steps:
            start_avg_time = time.perf_counter()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        pred = out.argmax(dim=-1)
        y = batch.y[:batch.batch_size].view(-1).to(torch.long)

        total_correct += int((pred == y).sum())
        total_examples += y.size(0)

    print(f'Average Inference Iteration Time (s/iter): {time.perf_counter() - start_avg_time:.6f}')

    return total_correct / total_examples


for epoch in range(1, 4):
    train()
    val_acc = test(val_loader, eval_steps=100)
    print(f'Val Acc: ~{val_acc:.4f}')

test_acc = test(test_loader)
print(f'Test Acc: {test_acc:.4f}')
