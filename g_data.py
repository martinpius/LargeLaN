import torch, os, requests
from timeit import default_timer as timer

BATCH_SIZE = 32
block_size = 256
embed_size = 384
n_layers = 6
dropout = 0.2
head_size = 64


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def time_fmt(t: float = timer())->float:
    h = int(t / (60 * 60))
    m = int(t % (60 * 60) / 60)
    s = int(t % 60)
    return f"hrs: {h:02}, mins: {m:>02}, secs: {s:>5.2f}"

def read_data():

    file_ = os.path.join(os.path.dirname("__file__"), "input.txt")
    if not os.path.exists(file_):
        path_ = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(file_, "w") as f:
            f.write(requests.get(path_).text)

    with open(file_, "r") as f:
        data = f.read()

    return data

def prep_data():

    df = read_data()
    chars_unq = sorted(list(set(df)))
    stoi = {s: i for i, s in enumerate(chars_unq)}
    itos = {i: s for s, i in stoi.items()}
    encode = lambda s: [stoi[k] for k in s]
    decode = lambda l: "".join(itos[i] for i in l)
    return encode, decode, chars_unq

data = read_data()
encode, decode, chars_unq = prep_data()
vocab_size = len(chars_unq)
EPOCHS = 50000
max_eval_iters = 200
eval_iters = 500

n = int(0.8 * len(data))
data = torch.tensor(encode(data), dtype = torch.long)
train, valid = data[:n], data[n:]

def get_batch(split):

    if split == "train":
        data = train
    else:
        data = valid
    IX = torch.randint(low = 0, high = (len(data)-block_size), size = (BATCH_SIZE,))

    xb = torch.stack([data[i: i + block_size] for i in IX])
    yb = torch.stack([data[i + 1: i + 1 + block_size] for i in IX])
    xb, yb = xb.to(device = device), yb.to(device = device)
    return xb, yb

if __name__ == "__main__":
    tic = timer()
    xb, yb = get_batch("train")

    for b in range(BATCH_SIZE):
        for t in range(block_size):
            context = xb[b, :t + 1]
            target = yb[b, t]
            print(f">>>> When the context is: {context}, The target is: {target}")
    print(f">>>> There are {len(chars_unq)} unique characters in the dataset which are:\
                  \n{''.join(chars_unq)}")
    print(f">>>> X batch shape: {xb.shape}, Y batch shape: {yb.shape}")
    toc = timer()
    print(f">>>> Time elapsed: {time_fmt(toc - tic)}")



