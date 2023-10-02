import torch
from torch import nn
from selfattention import SelfAttention
from g_data import head_size, embed_size, BATCH_SIZE, device, block_size, dropout

class MultHeadAttention(nn.Module):
    def __init__(self, embed_size, head_size)->None:
        super().__init__()
        num_heads = embed_size // head_size
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        '''
        x: shape ==> [B-> Batch size, T-> Block_size, C-> embed_size]
        This is an output from the embedding layers (token embedding + pos embedding)
        '''
        #B, T, C = x.shape
        multihead = [head(x) for head in self.heads] # shape == [B, T, head_size]
        out = torch.cat(multihead, dim = 2) # concatenate all heads: shape ==> [B, T, C = embed_size]
        out = self.dropout(self.projection(out))
        return out

if __name__ == "__main__":
    x = torch.rand(size = (BATCH_SIZE, block_size, embed_size), device = device)
    multihead = MultHeadAttention(embed_size, head_size).to(device = device)
    out = multihead(x)
    assert out.shape == (BATCH_SIZE, block_size, embed_size)

