# count approach trigram
import torch
import random

random.seed(12345)
raw_words = open('names.txt', 'r').read().splitlines()
training_words = []
dev_words = []
test_words = []
for word in raw_words:
  if random.random() <= 0.8:
    training_words.append(word)
  elif random.random() < 0.5:
    dev_words.append(word)
  else:
    test_words.append(word)


print(len(training_words)/(len(raw_words)))
print(len(dev_words)/len(raw_words))
print(len(test_words)/len(raw_words))

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(training_words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in training_words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1

P = (N+1).float()
P /= P.sum(1, keepdims=True)

log_likelihood = 0.0
n = 0

for w in dev_words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
print('Dev - Bigram')
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')

log_likelihood = 0.0
n = 0
for w in test_words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
print('Test - Bigram')
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')

chars = sorted(list(set(''.join(training_words))))
chars.append('.')
char_pairs = [i+j for i in chars for j in chars]
s2toi = {s:i for i,s in enumerate(char_pairs)}
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}

N = torch.zeros((len(char_pairs), 27), dtype=torch.int32)


for w in training_words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = s2toi[ch1+ch2]
    ix2 = stoi[ch3]
    N[ix1, ix2] += 1

P = (N+1).float()
P /= P.sum(1, keepdims=True)

log_likelihood = 0.0
n = 0

for w in dev_words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = s2toi[ch1+ch2]
    ix2 = stoi[ch3]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1

print('Dev - Trigram')
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')

log_likelihood = 0.0
n = 0

for w in test_words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = s2toi[ch1+ch2]
    ix2 = stoi[ch3]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1

print('Test - Trigram')
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')


P = (N+0.2).float()
P /= P.sum(1, keepdims=True)

log_likelihood = 0.0
n = 0

for w in dev_words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = s2toi[ch1+ch2]
    ix2 = stoi[ch3]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1

print('Dev - Trigram - Smoothing Test')
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')

log_likelihood = 0.0
n = 0

for w in test_words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
    ix1 = s2toi[ch1+ch2]
    ix2 = stoi[ch3]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1

print('Test - Trigram - Final Values')
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')
