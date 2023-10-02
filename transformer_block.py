import torch
from torch import nn
from multihead import MultHeadAttention
from feednets import FeedNets
from g_data import embed_size, head_size, BATCH_SIZE, block_size, device

class Transformer(nn.Module):
    '''
    Here we combine the attention modules and computations
    modules before the final logits: We also normalize the layers
    as we getting deeper
    '''
    def __init__(self, head_size, embed_size)->None:
        super().__init__()
        self.communications = MultHeadAttention(embed_size, head_size)
        self.computations = FeedNets(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        '''
        Here we construct a transformer's block with
        skip connections and layer normalization [Since the network is getting deeper]

        '''
        x = x + self.communications(self.ln1(x)) # Shape ==> [B, T, C]
        x = x + self.computations(self.ln2(x)) # Shape ==> [B, T, C]
        return x

if __name__ == "__main__":
    x = torch.randn(size = (BATCH_SIZE, block_size, embed_size), device = device)
    transformer = Transformer(head_size, embed_size).to(device = device)
    out = transformer(x)
    assert out.shape == (BATCH_SIZE, block_size, embed_size)

    