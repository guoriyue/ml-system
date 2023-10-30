"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT, DEBUG
import time, json

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 3 # number of samples to draw
num_warmup = 1 # how many warmups to do before benchmarking
max_new_tokens = 128 # number of tokens generated in each sample
temperature = 0.4 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
speculative_tokens = 3 # how many tokens should the draft model decode?
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
batch_size = 32
prompt_length = 64
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float32' if DEBUG else 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if DEBUG:
    torch.backends.cuda.matmul.allow_tf32 = False # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = False # allow tf32 on cudnn
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
else:
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
print(f'Running on {device} in {ptdtype}')
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.endswith('jsonl'):
    with open(start, 'r', encoding='utf-8') as f:
        lines = [json.loads(x)['prompt'] for x in f.readlines()]
    start_ids = [encode(x) for x in lines]
    print('NUM SATISFYING PROMPTS =', len([x for x in start_ids if len(x) > prompt_length]))
    x = torch.tensor([x[:prompt_length] for x in start_ids if len(x) > prompt_length][:batch_size], dtype=torch.long, device=device)
    print('RUNNING WITH BATCH SIZE =', x.size(0))
else:
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print(f'\n---------------\n\nRunning naive inference with top_k={top_k} and temperature={temperature:3f}\n\n---------------\n')

# benchmark naive generation
generations_naive = []
# warmup
print('Starting warmup')
with torch.no_grad():
    with ctx:
        for i in range(num_warmup):
            model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
print('Finished warmup')
# run
t0n = time.time()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            torch.manual_seed(k+1337) # we want consistency
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generations_naive.append(y[0].tolist())
            print(decode(generations_naive[-1]))
            print('---------------')
t1n  = time.time()


print(f'\n---------------\n\nRunning kv cache inference with top_k={top_k} and temperature={temperature:3f}\n\n---------------\n')

# benchmark speculative generation
generations_kv = []
# turn on the kv cache!
model.enable_kv(True)
print('Starting warmup')
with torch.no_grad():
    with ctx:
        for i in range(num_warmup):
            model.generate_kv(x, max_new_tokens, temperature=temperature, top_k=top_k)
print('Finished warmup')
# run
t0k = time.time()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            torch.manual_seed(k+1337) # we want consistency
            y = model.generate_kv(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generations_kv.append(y[0].tolist())
            print(decode(generations_kv[-1]))
            print('---------------')
t1k  = time.time()

if x.size(0) == 1:

    print(f'\n---------------\n\nRunning speculative inference with top_k={top_k}, temperature={temperature:3f}, and speculative_tokens={speculative_tokens}\n\n---------------\n')

    # benchmark speculative generation
    generations_spec = []
    # load speculative model
    print('Loading draft model')
    draft_model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
    draft_model.eval()
    draft_model.to(device)
    if compile:
        draft_model = torch.compile(draft_model) # requires PyTorch 2.0 (optional)
    draft_model.enable_kv(True)
    print('Finished loading draft model')
    # warmup
    print('Starting warmup')
    with torch.no_grad():
        with ctx:
            for i in range(num_warmup):
                model.generate_speculative(x, max_new_tokens, draft_model, temperature=temperature, top_k=top_k)
    print('Finished warmup')
    # run
    t0s = time.time()
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                torch.manual_seed(k+1337) # we want consistency
                y = model.generate_speculative(x, max_new_tokens, draft_model, temperature=temperature, top_k=top_k)
                generations_spec.append(y[0].tolist())
                print(decode(generations_spec[-1]))
                print('---------------')
    t1s  = time.time()

else:
    print('Skipping speculative inference because batch_size > 1')

tokens_generated = x.size(0) * num_samples * max_new_tokens
print(f'[NAIVE] Generated {tokens_generated} tokens in {t1n-t0n:3f} seconds for {tokens_generated/(t1n-t0n):3f} tok/sec.')
print(f'[KV]    Generated {tokens_generated} tokens in {t1k-t0k:3f} seconds for {tokens_generated/(t1k-t0k):3f} tok/sec.')
if x.size(0) == 1: print(f'[SPEC]  Generated {tokens_generated} tokens in {t1s-t0s:3f} seconds for {tokens_generated/(t1s-t0s):3f} tok/sec.')

all_matched = True
# generations_matching and generations_kv
num_matching = len([1 for i in range(len(generations_naive)) if generations_naive[i] == generations_kv[i]])
print(f'NAIVE and KV generations matched on {num_matching} positions')
if num_matching < len(generations_naive):
    all_matched = False

if x.size(0) == 1:
    num_matching = len([1 for i in range(len(generations_naive)) if generations_naive[i] == generations_spec[i]])
    print(f'NAIVE and SPEC generations matched on {num_matching} positions')
    if num_matching < len(generations_naive):
        all_matched = False
if all_matched:
    print('All generations matched :)')
else:
    print('NOT all generations matched :(')