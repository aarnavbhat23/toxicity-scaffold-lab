import torch
from torch import nn
from chemistry import ATOM_DIM, BOND_DIM


class ToxicityNetwork(nn.Module):
    """Three bond-aware message-passing layers and twelve assay heads."""
    def __init__(self, tasks=12, width=64):
        super().__init__()
        self.embed = nn.Linear(ATOM_DIM, width)
        self.messages = nn.ModuleList([nn.Linear(width+BOND_DIM,width) for _ in range(3)])
        self.updates = nn.ModuleList([nn.GRUCell(width,width) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,tasks))

    def forward(self, atoms, edges, bonds, membership, count):
        hidden = torch.relu(self.embed(atoms))
        source, destination = edges
        for message, update in zip(self.messages, self.updates):
            values = torch.relu(message(torch.cat([hidden[source],bonds],1)))
            aggregate = torch.zeros_like(hidden).index_add(0,destination,values)
            degree = torch.zeros((len(hidden),1), device=hidden.device).index_add(
                0,destination,torch.ones((len(destination),1),device=hidden.device)).clamp_min(1)
            hidden = update(aggregate/degree,hidden)
        pooled = torch.zeros((count,hidden.shape[1]), device=hidden.device).index_add(0,membership,hidden)
        sizes = torch.bincount(membership,minlength=count).unsqueeze(1).clamp_min(1)
        return self.head(pooled/sizes)


def masked_bce(logits, targets):
    observed = torch.isfinite(targets)
    if not observed.any():
        raise ValueError('No observed assay targets')
    return nn.functional.binary_cross_entropy_with_logits(logits[observed],targets[observed])
