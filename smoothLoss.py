from g_data import max_eval_iters, get_batch, device
import torch, sys
from nanogpt import NanoGPT

model = NanoGPT().to(device = device)

@torch.no_grad()
def estimate_loss():
    cache = {}
    losses = torch.zeros(max_eval_iters)
    model.eval()
    for split in ["train","valid"]:
        for k in range(max_eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        cache[split] = losses.mean()
    model.train()
    return cache

if __name__ == "__main__":
    cache = estimate_loss()
    print(f">>>> Loss for the training set: {cache['train']:.4f}, Loss for validation set: {cache['valid']:.4f}")
    sys.exit()
    

