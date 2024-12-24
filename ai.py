from __future__ import annotations

import base64, math, regex
from collections import defaultdict
from typing import Any, Callable, List, Tuple

import mlx.core as mx

def bpe_encode(mergeable_ranks: dict[bytes, int], input: bytes) -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        if min_rank is None:
            break

        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

    return [mergeable_ranks[part] for part in parts]

class Tokenizer:
    def __init__(self, tokenizer_model: str) -> None:
        self._pat = regex.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

        with open(tokenizer_model) as f:
            lines = f.read()
            self.mergeable_ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in lines.splitlines() if line)}

        self._decoder = {token: token_bytes for token_bytes, token in self.mergeable_ranks.items()}

        self.vocab = self.mergeable_ranks
        self.bos_token_id = 128000 # <|begin_of_text|>
        self.eos_token_id = 128001 # <|end_of_text|>

    def encode(self, text: str) -> list[int]:
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            word_bytes = word.encode("utf-8")
            word_tokens = bpe_encode(self.mergeable_ranks, word_bytes)
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return b"".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

def tree_map(fn: Callable, tree: Any, *rest: Any):
    if isinstance(tree, (list, tuple)):
        TreeType = type(tree)
        return TreeType(tree_map(fn, child, *(r[i] for r in rest)) for i, child in enumerate(tree))

    if isinstance(tree, dict):
        return {k: tree_map(fn, child, *(r[k] for r in rest)) for k, child in tree.items()}

    return fn(tree, *rest)

def tree_flatten(tree: Any, prefix: str = ""):
    flat_tree = []

    if isinstance(tree, (list, tuple)):
        for i, t in enumerate(tree):
            flat_tree.extend(tree_flatten(t, f"{prefix}.{i}"))
        return flat_tree

    if isinstance(tree, dict):
        for k, t in tree.items():
            flat_tree.extend(tree_flatten(t, f"{prefix}.{k}"))
        return flat_tree

    return [(prefix[1:], tree)]

def tree_unflatten(tree: List[Tuple[str, Any]]):
    if len(tree) == 1 and tree[0][0] == "":
        return tree[0][1]

    try:
        int(tree[0][0].split(".", maxsplit=1)[0])
        is_list = True
    except ValueError:
        is_list = False

    children = defaultdict(list)
    for key, value in tree:
        current_idx, *next_idx = key.split(".", maxsplit=1)
        next_idx = "" if not next_idx else next_idx[0]
        children[current_idx].append((next_idx, value))

    if is_list:
        keys = sorted((int(idx), idx) for idx in children.keys())
        l = []
        for i, k in keys:
            l.extend([{} for _ in range(i - len(l))])
            l.append(tree_unflatten(children[k]))
        return l
    else:
        return {k: tree_unflatten(v) for k, v in children.items()}

class Module(dict):
    __call__: Callable

    def __init__(self):
        self._no_grad = set()

    @property
    def state(self):
        return self

    def __getattr__(self, key: str):
        if (value := self.get(key, None)) is not None:
            return value
        else:
            super(Module, self).__getattribute__(key)

    def __setattr__(self, key: str, val: Any):
        if isinstance(val, (mx.array, dict, list, tuple)):
            if hasattr(self, key) and key not in self:
                delattr(self, key)
            self[key] = val
        else:
            super(Module, self).__setattr__(key, val)
            self.pop(key, None)

    @staticmethod
    def load(file: str, shape):
        with open(file, "rb") as f:
            return mx.array(f.read()).view(dtype=mx.bfloat16).reshape(shape)

    def load_weights(self, weights: str):
        _ = [(name, Module.load(weights + '/' + name, data.shape)) for name, data in tree_flatten(self.parameters())]

        self.update(tree_unflatten(_))
        return self

    def save_weights(self, weights: str):
        for name, data in tree_flatten(self.parameters()):
            with open(weights + '/' + name, "wb") as f:
                f.write(data)

        return self

    @staticmethod
    def valid_parameter_filter(module, key, value):
        return isinstance(value, (dict, list, mx.array)) and not key.startswith("_")

    @staticmethod
    def trainable_parameter_filter(module, key, value):
        return (Module.valid_parameter_filter(module, key, value) and key not in module._no_grad)

    def filter(self, filter_fn: Callable[[Module, str, Any], bool]):
        return {k: _unwrap(self, k, v, filter_fn) for k, v in self.items() if filter_fn(self, k, v)}

    def parameters(self):
        return self.filter(self.valid_parameter_filter)

    def trainable_parameters(self):
        return self.filter(self.trainable_parameter_filter)

    def update(self, parameters: dict):
        def apply(dst, parameters):
            if isinstance(parameters, dict):
                for k in parameters:
                    if k in dst:
                        current_value = dst[k]
                        new_value = parameters[k]
                        if isinstance(current_value, mx.array):
                            dst[k] = new_value
                        elif isinstance(current_value, Module):
                            current_value.update(new_value)
                        elif isinstance(current_value, (dict, list)):
                            apply(current_value, new_value)
            elif isinstance(parameters, list):
                for i in range(len(parameters)):
                    current_value = dst[i]
                    new_value = parameters[i]
                    if isinstance(current_value, mx.array):
                        dst[i] = new_value
                    elif isinstance(current_value, Module):
                        current_value.update(new_value)
                    elif isinstance(current_value, (dict, list)):
                        apply(current_value, new_value)

        apply(self, parameters)
        return self

def _unwrap(model, value_key, value, filter_fn):
    if isinstance(value, Module):
        return {k: _unwrap(value, k, v, filter_fn) for k, v in value.items() if filter_fn(value, k, v)}

    if isinstance(value, dict):
        nd = {}
        for k, v in value.items():
            tk = f"{value_key}.{k}"
            nd[k] = (_unwrap(model, tk, v, filter_fn) if filter_fn(model, tk, v) else {})
        return nd

    if isinstance(value, list):
        nl = []
        for i, vi in enumerate(value):
            tk = f"{value_key}.{i}"
            nl.append(_unwrap(model, tk, vi, filter_fn) if filter_fn(model, tk, vi) else {})
        return nl

    return value

class ModelArgs:
    hidden_size: int = 3072
    num_hidden_layers: int = 7
    intermediate_size: int = 8192
    num_attention_heads: int = 24
    vocab_size: int = 128256
    num_key_value_heads: int = 8

class RoPE(Module):
    def __init__(self, dims: int, scale: float = 1.0):
        super().__init__()
        self.dims = dims
        self.scale = scale

        base = 500000.0

        factor = 32.0
        low_freq_factor = 1.0
        high_freq_factor = 4.0
        old_context_len = 8192

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, self.dims, 2) / self.dims)
        wavelens = 2 * mx.pi * freqs

        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(x, self.dims, traditional=False, base=None, scale=self.scale, offset=offset, freqs=self._freqs)

class KVCache:
    def __init__(self, head_dim, n_kv_heads):
        self.n_kv_heads = n_kv_heads
        self.k_head_dim = self.v_head_dim = head_dim
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        return self.keys, self.values

class Linear(Module):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(output_dims, input_dims), dtype=mx.bfloat16)

    def __call__(self, x: mx.array):
        return x @ self.weight.T

class Attention(Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = math.sqrt(1.0 / head_dim)

        self.q_proj = Linear(dim, n_heads * head_dim)
        self.k_proj = Linear(dim, n_kv_heads * head_dim)
        self.v_proj = Linear(dim, n_kv_heads * head_dim)
        self.o_proj = Linear(n_heads * head_dim, dim)

        self.rope = RoPE(dims=head_dim)

    def __call__(self, e: mx.array, mask: mx.array, cache=None):
        B, L, D = e.shape

        queries, keys, values = self.q_proj(e), self.k_proj(e), self.v_proj(e)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

def silu(x):
    return x * mx.sigmoid(x)

class FeedForward(Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size

        self.gate_proj = Linear(dim, hidden_dim)
        self.down_proj = Linear(hidden_dim, dim)
        self.up_proj = Linear(dim, hidden_dim)

    def __call__(self, e):
        return self.down_proj(silu(self.gate_proj(e)) * self.up_proj(e))

class RMSNorm(Module):
    def __init__(self, dims: int, eps: float = 0.00001):
        super().__init__()
        self.weight = mx.ones((dims,), dtype=mx.bfloat16)
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)

class TransformerBlock(Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = FeedForward(args)
        self.input_layernorm = RMSNorm(args.hidden_size)
        self.post_attention_layernorm = RMSNorm(args.hidden_size)
        self.args = args

    def __call__(self, e: mx.array, mask: mx.array, cache=None):
        r = self.self_attn(self.input_layernorm(e), mask, cache)
        e = e + r
        r = self.mlp(self.post_attention_layernorm(e))
        return e + r

class Embedding(Module):
    def __init__(self, num_embeddings: int, dims: int):
        super().__init__()
        scale = math.sqrt(1.0 / dims)
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(num_embeddings, dims), dtype=mx.bfloat16)

    def __call__(self, x):
        return self.weight[x]

    def as_linear(self, x):
        return x @ self.weight.T

def create_additive_causal_mask(T: int, offset: int):
    rinds = mx.arange(offset + T)
    linds = mx.arange(offset, offset + T) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9

def create_attention_mask(e: mx.array, cache=None):
    T = e.shape[1]
    if T > 1:
        if cache is not None and cache[0] is not None:
            c = cache[0]
            offset = c.offset
        else:
            offset = 0
        mask = create_additive_causal_mask(T, offset)
        mask = mask.astype(e.dtype)
    else:
        mask = None
    return mask

class LlamaModel(Module):
    def __init__(self):
        super().__init__()
        self.args = args = ModelArgs()
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size)

    def __call__(self, inputs: mx.array, cache=None):
        e = self.embed_tokens(inputs)

        mask = create_attention_mask(e, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            e = layer(e, mask, cache=c)

        e = self.norm(e)
        return self.embed_tokens.as_linear(e)

class Model(Module):
    def __init__(self, weights=None):
        super().__init__()
        self.model = LlamaModel()

        if weights is not None:
            self.load_weights(weights)

        mx.eval(self.parameters())

    def __call__(self, inputs: mx.array, cache=None):
        return self.model(inputs, cache)

    @property
    def args(self):
        return self.model.args

    @property
    def layers(self):
        return self.model.layers

class Detokenizer:
    def __init__(self, tokenizer):
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = bytes([])
        self.text = ""
        self.tokens = []

    def add_token(self, token):
        v = self.tokenmap[token]
        if v[0] == 32:
            current_text = bytearray(c for c in self._unflushed).decode("utf-8")
            self.text += current_text
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        current_text = bytearray(c for c in self._unflushed).decode("utf-8")
        self.text += current_text
        self._unflushed = bytes([])

    @property
    def last_segment(self):
        text = self.text
        if text:
            segment = text[self.offset :]
            self.offset = len(text)
            return segment
        return ""

def _generate(model: Module, prompt: mx.array):
    head_dim = model.args.hidden_size // model.args.num_attention_heads
    kv_heads = [model.args.num_key_value_heads] * len(model.layers)
    cache = [KVCache(head_dim, n) for n in kv_heads]

    def _step(y):
        logits = model(y[None], cache=cache)

        return mx.argmax(logits[:, -1, :], axis=-1)

    y = prompt

    while y.size > 1:
        model(y[:1][None], cache=cache)
        mx.eval([c.state for c in cache])
        y = y[1:]

    y = _step(y)
    mx.eval(y)
    while True:
        next_y = _step(y)
        mx.eval(next_y)
        yield y.item()
        y = next_y

def generate(model: Module, tokenizer, detokenizer, prompt: str, max_tokens: int = 256):
    print(prompt, end="", flush=True)

    prompt_tokens = mx.array(tokenizer.encode(prompt))

    detokenizer.reset()

    for token, _ in zip(_generate(model, prompt_tokens), range(max_tokens)):
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)

        print(detokenizer.last_segment, end="", flush=True)

    detokenizer.finalize()

    print(detokenizer.last_segment, flush=True)

class Optimizer:
    def __init__(self):
        self._initialized = False
        self._state = {"step": mx.array(0, mx.uint64)}

    def update(self, model: Module, gradients: dict):
        model.update(self.apply_gradients(gradients, model))

    def init(self, parameters: dict):
        def update_state(params, state):
            if isinstance(params, (list, tuple)):
                state = list(state)
                for i in range(len(state)):
                    state[i] = update_state(params[i], state[i])
                if len(state) != len(params):
                    state.extend(tree_map(lambda x: {}, params[len(state) :]))
                return type(params)(state)
            elif isinstance(params, dict):
                for k, v in params.items():
                    if k not in state:
                        state[k] = tree_map(lambda x: {}, v)
                    else:
                        state[k] = update_state(v, state[k])
                return state
            else:
                return state

        update_state(parameters, self._state)
        tree_map(lambda p, s: s or self.init_single(p, s), parameters, self._state)
        self._initialized = True

    def init_single(self, parameter: mx.array, state: dict):
        raise NotImplementedError()

    def apply_gradients(self, gradients: dict, parameters: dict):
        if not self._initialized:
            self.init(gradients)

        self.state["step"] = self.step + 1

        return tree_map(self.apply_single, gradients, parameters, self.state)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        raise NotImplementedError()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: dict):
        self._initialized = False
        self._state = state

    @property
    def step(self):
        return self.state["step"]

    @property
    def learning_rate(self):
        return self.state["learning_rate"]

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self.state["learning_rate"] = mx.array(learning_rate)

class AdamW(Optimizer):
    def __init__(self, learning_rate: float, betas: List[float] = [0.9, 0.999], eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__()

        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def init_single(self, parameter: mx.array, state: dict):
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        eps = self.eps

        m = state["m"]
        v = state["v"]
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * mx.square(gradient)
        state["m"] = m
        state["v"] = v

        return parameter * (1 - lr * self.weight_decay) - lr * m / (mx.sqrt(v) + eps)

def cross_entropy(logits: mx.array, targets: mx.array, axis: int = -1):
    targets_as_probs = targets.ndim == logits.ndim

    def _drop_dim(shape, axis):
        shape = list(shape)
        shape.pop(axis)
        return tuple(shape)

    if (targets_as_probs and targets.shape != logits.shape) or (not targets_as_probs and targets.shape != _drop_dim(logits.shape, axis)):
        raise ValueError(f"Targets shape {targets.shape} does not match logits shape {logits.shape}.")

    if targets_as_probs:
        score = mx.sum(logits * targets, axis=axis)
    else:
        score = mx.take_along_axis(logits, targets[..., None], axis).squeeze(-1)

    return mx.logsumexp(logits, axis=axis) - score

def _cross_entropy(logits: mx.array, targets: mx.array, axis: int = -1):
    if (targets.ndim != logits.ndim):
        raise ValueError(f"Targets ndim {targets.ndim} does not match logits ndim {logits.ndim}.")

    if (targets.shape != logits.shape):
        raise ValueError(f"Targets shape {targets.shape} does not match logits shape {logits.shape}.")

    return mx.logsumexp(logits, axis=axis) - mx.sum(logits * targets, axis=axis)

def cross_entropy_loss(model, inputs, targets, lengths):
    logits = model(inputs)
    logits = logits.astype(mx.float32)

    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    ce = cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks

def value_and_grad(model: Module, fn: Callable):
    def inner_fn(params, *args, **kwargs):
        model.update(params)
        return fn(*args, **kwargs)

    value_grad_fn = mx.value_and_grad(inner_fn)

    def wrapped_value_grad_fn(*args, **kwargs):
        value, grad = value_grad_fn(model.trainable_parameters(), *args, **kwargs)
        return value, grad

    return wrapped_value_grad_fn

def main():
    model = Model("bf16")
    tokenizer = Tokenizer("./tokenizer.model")
    detokenizer = Detokenizer(tokenizer)

    generate(model, tokenizer, detokenizer, prompt="Már nem volt fiatal, de még")

    e = tokenizer.encode("Már nem volt fiatal, de még elég jól bírta magát; ismerték és félték a nádas lakói, de még azon túl is, közelben-távolban, minden négylábú lény. Látása nem romlott, s ha ezerméteres magasságból kiszemelte zsákmányát, úgy csapott le rá, mint egy kalapács, mely egyetlen ütéssel veri be a szöget. És így, viruló korában, ereje teljében, két lassú szárnycsapás között egyszer csak megállt a szíve. De nem mertek előbújni sem a nyulak, sem az ürgék, sem a környező falvak baromfiai, mert ő ott lebegett ezer méter magasban, kiterjesztett szárnyával, fenyegető mozdulatlanságban túlélve a halált még két vagy három perccel, míg el nem állt a szél.")

    x = mx.array([e[:-1]])
    y = mx.array([e[1:]])
    z = mx.array([len(e)])

    optimizer = AdamW(learning_rate=0.00001, betas=[0.9, 0.97], eps=0.00001, weight_decay=0.0)
    state = [model.state, optimizer.state]
    loss_value_and_grad = value_and_grad(model, cross_entropy_loss)

    for step in range(20):
        (loss, ntoks), grad = loss_value_and_grad(model, x, y, z)
        optimizer.update(model, grad)
        mx.eval(state, loss, ntoks)
        print(f"loss: {loss.item()}")

    generate(model, tokenizer, detokenizer, prompt="Már nem volt fiatal, de még")

if __name__ == "__main__":
    main()
