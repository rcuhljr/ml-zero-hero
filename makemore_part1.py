# count approach trigram
import torch

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
chars.append('.')
char_pairs = [i+j for i in chars for j in chars]
s2toi = {s:i for i,s in enumerate(char_pairs)}
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}

N = torch.zeros((len(char_pairs), 27), dtype=torch.int32)


for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = s2toi[ch1+ch2]
    ix2 = stoi[ch3]
    N[ix1, ix2] += 1

P = (N+1).float()
P /= P.sum(1, keepdims=True)
print(P[0].sum())

g = torch.Generator().manual_seed(2147483647)

for i in range(10):

  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 26:
      break
  print(''.join(out))

log_likelihood = 0.0
n = 0

for w in words:
#for w in ["andrejq"]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = s2toi[ch1+ch2]
    ix2 = stoi[ch3]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')


import torch.nn.functional as F

xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = s2toi[ch1+ch2]
    ix2 = stoi[ch3]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(21474836471)
W = torch.randn((27*27, 27), generator=g, requires_grad=True)
print("Original Loss")
for k in range(100):

  # forward pass
  # xenc = F.one_hot(xs, num_classes=27*27).float() # input to the network: one-hot encoding
  # logits = xenc @ W # predict log-counts
  logits = W.index_select(0, xs)
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.0005*(W**2).mean()
  if k % 10 == 0:
    print(loss.item())

  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()

  # update
  W.data += -400 * W.grad

print(loss.item())

print("Cross Entropy")
g = torch.Generator().manual_seed(21474836471)
W = torch.randn((27*27, 27), generator=g, requires_grad=True)

for k in range(100):

  # forward pass
  # xenc = F.one_hot(xs, num_classes=27*27).float() # input to the network: one-hot encoding
  # logits = xenc @ W # predict log-counts
  logits = W.index_select(0, xs)
  loss = F.cross_entropy(logits, ys)
  if k % 10 == 0:
    print(loss.item())

  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()

  # update
  W.data += -400 * W.grad

print(loss.item())
