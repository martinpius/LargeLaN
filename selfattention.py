import torch
from g_data import device, block_size, embed_size, head_size, dropout, BATCH_SIZE

class SelfAttention(torch.nn.Module):

    def __init__(self, head_size)->None:
        super().__init__()

        self.query = torch.nn.Linear(embed_size, head_size, bias = False)
        self.keys = torch.nn.Linear(embed_size, head_size, bias = False)
        self.values = torch.nn.Linear(embed_size, head_size, bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(size = (block_size, block_size))))
        self.dropout = torch.nn.Dropout(p = dropout)
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        '''
        x: shape ==> [B-> Batch, T-> block size, C-> embed_size]
        This is an output from the tokens embedding + possition embedding layers
        '''
        B, T, C = x.shape

        Q = self.query(x) # shape ==> [B, T, head_size]
        K = self.keys(x) # shape ==> [B, T, head_size]
        V = self.values(x) # shape ==> [B, T, head_size]

        Wei = Q @ K.transpose(-2, -1) # shape ==> [B, T, T] : Fetching the affinities between the target and the inputs
        Wei = Wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # Masking future communications
        Wei = torch.nn.functional.softmax(Wei, dim = -1) # Compute the proba distribution for the affinities
        Wei = self.dropout(Wei)
        out = Wei @ V # Shape ==> [B, T, head_size] # Single head attention
        return out

if __name__ == "__main__":
    x = torch.randn(size = (BATCH_SIZE, block_size, embed_size), device = device)
    selfattn = SelfAttention(head_size = head_size).to(device = device)
    out = selfattn(x)
    assert out.shape == (BATCH_SIZE, block_size, head_size)

