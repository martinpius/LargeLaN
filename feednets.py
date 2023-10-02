import torch
from torch import nn
from g_data import embed_size, BATCH_SIZE, device, dropout, block_size

class FeedNets(nn.Module):

    def __init__(self, embed_size)->None:
        super().__init__()
        self.feednet = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(p = dropout)
        )
    def forward(self, x: torch.Tensor)->torch.Tensor:
        '''
        x: output from the attention block, before computing the final logits
        Shape ==> [B, T, C]
        '''
        return self.feednet(x)

if __name__ == "__main__":
    x = torch.randn(size = (BATCH_SIZE, block_size, embed_size), device = device)
    feednet = FeedNets(embed_size = embed_size).to(device = device)
    out = feednet(x)
    assert out.shape == (BATCH_SIZE, block_size, embed_size)
    