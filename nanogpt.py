import torch
from torch import nn
from transformer_block import Transformer
from g_data import embed_size, block_size, n_layers, device, vocab_size, head_size, get_batch,decode

class NanoGPT(nn.Module):
    '''
    We use our transomer block to construct a nano-GPT model
    i.e., we stack 6 transformer's block , we also use layer
    normalization as we get deeper
    '''

    def __init__(self):

        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size)
        self.position_embedding = nn.Embedding(num_embeddings = block_size, embedding_dim = embed_size)
        self.nano_gpt = nn.Sequential(*[Transformer(head_size = head_size, embed_size = embed_size) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.fc = nn.Linear(embed_size, vocab_size) 
    
    def forward(self, IX: torch.Tensor, target: torch.Tensor = None)-> torch.Tensor:
        '''
        IX: inputs tokens [tokenized texts]==> shape: [B-> Batch, T->Block size]
        target: target tokens[tokenized text]==> shape: [B, T]
        '''
        _, T = IX.shape
        tkn_embed = self.token_embedding(IX) # shape ==> [B, T, C => embed_size]
        pos_tokens = torch.arange(T, device = device) # shape ==> [block_size]
        pos_embed = self.position_embedding(pos_tokens) # shape ==> [T, C]
        x = tkn_embed + pos_embed # shape ==> [B, T, C]
        x = self.nano_gpt(x) # shape ==> [BTC]
        x = self.layer_norm(x)
        logits = self.fc(x) # shape ==> [B, T, C = vocab_size]

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape # unpacking the dimensions to reshape inputs of CE loss
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits, target)
        return logits, loss
    
    def generator(self, idx: torch.Tensor, max_tokens: int)->torch.Tensor:

        '''
        This module generate new tokens and concatenate with the
        previous tokens
        idx: shape ==> [B, T]
        '''

        for _ in range(max_tokens):
            idx_clipped = idx[:, -block_size:] # clip the block size to the maximum of block_size used in the train
            # Run the forward pass, We don need the loss since we are doing inference
            logits, _ = self(idx_clipped)
            # get the last token as a prediction
            logits = logits[:, -1, :] # shape ==> [B, C=> vocab_size]
            probs = torch.nn.functional.softmax(logits, dim = 1) # shape==>[B, C]: get the prob distribution for the possible next token
            idx_new = torch.multinomial(input = probs, num_samples = 1) # shape ==> [B, 1] # sample the new token according to the multinomial dist
            idx = torch.cat([idx, idx_new], dim = 1) # concatenate newly generated token--> shape: [B, T + 1 etc]
        return idx

if __name__ == "__main__":
    xb, yb = get_batch("train")
    nano_gpt = NanoGPT().to(device = device)
    logits, loss = nano_gpt(xb, yb)
    print(f">>>> Loss: {loss.item():.4f}, Logits shape: {logits.shape}")
    text_codes = nano_gpt.generator(torch.zeros(size = (1,1), dtype = torch.long, device = device), 1000)[0].tolist()
    print(f">>>> Random generated texts is:\n\n {decode(text_codes)}")