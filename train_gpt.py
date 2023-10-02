from g_data import get_batch, time_fmt, EPOCHS, eval_iters, device, decode
from timeit import default_timer as timer
from nanogpt import NanoGPT
from smoothLoss import estimate_loss
from tqdm.auto import tqdm
import torch

model = NanoGPT().to(device = device)
LR = 3e-4
optimizer = torch.optim.AdamW(params = model.parameters(), lr = LR)
print(f">>>> {model.__class__.__name__} has {sum([p.numel() for p in model.parameters()]):,} trainable parameters")

def my_trainer():

    tic = timer()
    for epoch in tqdm(range(EPOCHS)):
        if epoch % eval_iters == 0:
            cache = estimate_loss()
            print(f">>>> Epoch: {epoch + 1 if epoch == 0 else epoch}, \
                Evaluation Loss on the training data: {cache['train']:.4f}, \
                    Evaluation Loss on the validation data: {cache['valid']:.4f}")
        
        xb, yb = get_batch("train")

        model.train()
        _, loss = model(xb, yb)
        if epoch % eval_iters == 0:
            print(f">>>> Epoch: {epoch + 1 if epoch == 0 else epoch}, Train Loss: {loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    tokens = model.generator(idx = torch.zeros(size = (1,1), dtype = torch.long), max_tokens = 1000)[0].tolist()
    print(f">>>> Generated texts for the trained nanoGPT model:\n {decode(tokens)}")
    toc = timer()
    print(f">>>> Total time elapsed for training the model for {EPOCHS} epochs:\n {time_fmt(toc - tic)}")

if __name__ == "__main__":
    print(f">>>> Training: Please wait...................................................................................................................")
    my_trainer()


