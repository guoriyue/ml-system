"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

# For assignment #2, we've preset some flags that increase numerical determinism. Control them with this flag!
DEBUG = False

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and not DEBUG
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        
        ### Initialize KV cache with up to a maximum sequence length. Also, keep track of how much of it is filled in self.seq_pos.
        batch_dim = 32 # the maximum batch dimension we'll support. This is because T4 has smol mem.
        seq_dim = 512 # the maximum sequence length we'll support
        self.seq_pos = 0 # disabled by default
        self.kv_enabled = False # disabled by default

        ### YOUR CODE HERE

        # initialize a register buffer to store the kv cache
        # you will need to enter shapes of the size of the kv cache
        # hint: the first dimension is '2' because we need to store both the key and the value for each token in the sequence
        # so TODO = (2, ?, ?, ?, ?)
        self.register_buffer("kv_cache", torch.empty(( 
            2, batch_dim, self.n_head, seq_dim, self.n_embd // self.n_head
        ), dtype=torch.bfloat16), persistent=False)

        ### END YOUR CODE HERE


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        print("k.shape, q.shape, v.shape", k.shape, q.shape, v.shape)
        print("self.kv_cache.shape", self.kv_cache.shape)
        print("B, T, C", B, T, C)
        ### Set KV cache if it exists:
        if self.kv_enabled:
            ### YOUR CODE HERE
            
            # set the new seq_pos
            self.seq_pos = T

            # update the kv cache
            self.kv_cache[:, :B, :, :self.seq_pos, :] = torch.cat((k, v), dim=0)

            ### END YOUR CODE HERE

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
    def decode(self, x): # note: x now only includes the tokns *after* the KV cache
        if not self.kv_enabled:
            raise ValueError("KV cache is not enabled!")
        
        B, T, C = x.size()

        # Use both the KV cache and the input to run MHA! Make sure to save new KV projections into the cache, too!
        ### YOUR CODE HERE

        # compute q, k, v projections of x
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # update the kv cache with the additional input
        new_kv = torch.cat((k, v), dim=0)
        self.kv_cache[:, :B, :, self.seq_pos:self.seq_pos + T, :] = new_kv

        # compute manual implmentation of attention using the kv cache
        k_concat = torch.cat((self.kv_cache[0, :B, :, :self.seq_pos, :], k), dim=2)
        v_concat = torch.cat((self.kv_cache[1, :B, :, :self.seq_pos, :], v), dim=2)

        att = (q @ k_concat.transpose(-2, -1)) * (1.0 / math.sqrt(k_concat.size(-1)))
        att = att.masked_fill(self.bias[:, :, :att.size(2), :att.size(3)] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v_concat  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # compute output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))

        # update the position
        self.seq_pos += T

        return y
        ### END YOUR CODE HERE

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def decode(self, x):
        x = x + self.attn.decode(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    # Some scaffolding from the 229s teaching staff.
    @property # this is just a convenience function to make accessing easier
    def seq_pos(self):
        return self.transformer.h[0].attn.seq_pos
    @seq_pos.setter
    def seq_pos(self, value):
        for block in self.transformer.h:
            block.attn.seq_pos = value
    def enable_kv(self, use_kv=True):
        self.seq_pos = 0 if use_kv else None
        self.kv_enabled = use_kv
        for block in self.transformer.h:
            block.attn.kv_enabled = use_kv

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x)
            loss = None

        return logits, loss
    
    def decode(self, idx):
        device = idx.device
        b, t = idx.size()
        assert self.seq_pos + t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Run decoding on the GPT model itself. Make sure you set the right position embeddings!

        ### YOUR CODE HERE

        # create token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(self.seq_pos, self.seq_pos + t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)


        # perform decoding
        for block in self.transformer.h:
            x = block(x)
        
        # apply transformer layer normalization
        x = self.transformer.ln_f(x)
        

        # apply the language model head
        logits = self.lm_head(x)

        # return the output
        self.seq_pos += t
        return logits

        ### END YOUR CODE HERE

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                if top_k == 1:
                    idx = torch.cat((idx, torch.argmax(logits, dim=-1).reshape((-1,1))), dim=1) # using argmax here preserves RNG state
                    continue
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution with temperature
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_kv(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        # Generate the full KV cache and decode the first token

        new_tokens = torch.empty((idx.size(0), 0))

        ### YOUR CODE HERE

        # run the idx through the forward pass to get the logits
        logits, _ = self.forward(idx) 
        
        # pluck the logits at the final position and scale by desired temperature
        logits_to_use = logits[:, -1, :] / temperature
        
        # optionally crop the logits to only the top k options
        if top_k == 1:
            idx_next = torch.argmax(logits_to_use, dim=-1).unsqueeze(-1)
        else:
            logits_to_use = torch.topk(logits_to_use, top_k, dim=-1).values
            probs = F.softmax(logits_to_use, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)


        # save the new tokens
        new_tokens = torch.cat((new_tokens, idx_next), dim=1)
        
        ### END YOUR CODE HERE

        for _ in range(max_new_tokens-1):
            # Now decode using the KV cache

            ### YOUR CODE HERE

            # decode from the new tokens to get the logits
            logits, _ = self.decode(torch.cat((idx, new_tokens), dim=1))

            # pluck the logits at the final position and scale by desired temperature
            logits_to_use = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k == 1:
                idx_next = torch.argmax(logits_to_use, dim=-1).unsqueeze(-1)
            else:
                logits_to_use = torch.topk(logits_to_use, top_k, dim=-1).values
                probs = F.softmax(logits_to_use, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # store the new tokens
            new_tokens = torch.cat((new_tokens, idx_next), dim=1)

            ### END YOUR CODE HERE

        return torch.cat((idx, new_tokens), dim=1)

    @torch.no_grad()
    def generate_speculative(self, idx, max_new_tokens, draft_model, temperature=1.0, top_k=None, num_speculative=4):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        self.enable_kv(False) # disable KV cache for the main model -- it's not worth the effort

        # note: making speculative decoding work with batch_size>1 is beyond this assignment's scope, because the
        # tensors rapidly become ragged. So, you can assume that batch_size=1 for this part.
        if idx.size(0) != 1:
            raise ValueError("speculative decoding only works with batch size 1")
        idx_length_original = idx.size(1)
        
        loop_counter = 0
        while idx.size(1) < idx_length_original+max_new_tokens:
            loop_counter += 1

            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size-num_speculative else idx[:, -self.config.block_size-num_speculative:]

            # Generate speculative tokens from the draft model. Then, run it through the main model.
            # Be sure to set top_k=1, otherwise you'll pollute the RNG! (Which will make your code harder to debug.)

            ### YOUR CODE HERE

            # use the draft_model to generate speculative tokens
            idx_speculative, _ = draft_model.generate(idx_cond, num_speculative, temperature, top_k=1)
            idx_speculative = torch.cat((idx_cond, idx_speculative), dim=1)

            # obtain the logits from the main model by passing in the idx_speculative
            all_logits, _ = self.forward(idx_speculative)
            
            ### END YOUR CODE HERE

            # Step through the predictions of the main model, sampling, and check whether they match the next token. Stop upon mismatch.

            ### YOUR CODE HERE

            # iterate from the end position of idx_cond (prefix sequence) to the end position of idx_speculative (generated sequence)
            for i in range(idx_cond.size(1), idx_speculative.size(1)):
                # pluck the logits at the current position and scale by desired temperature
                logits = all_logits[:, i, :] / temperature
                
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    logits = torch.topk(logits, top_k, dim=-1).values

                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)

                # sample from the distribution with temperature
                idx_next = torch.multinomial(probs, num_samples=1)

                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)

                # end the loop if the next token does not match the next token in idx_speculative
                if idx_next.item() != idx_speculative[:, i].item():
                    break
                
            ### END YOUR CODE HERE
                
        print(f"speculative decoding ran for {loop_counter} iterations")
        return idx[:,:idx_length_original+max_new_tokens]
