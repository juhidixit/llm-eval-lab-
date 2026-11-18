"""In-process GGUF models via llama-cpp-python (``pip install llm-eval-lab[local]``).

Runs open-weight models (Qwen, Llama, Gemma, ...) on a laptop CPU with no server and no API key.
Unlike most local servers it returns exact first-token log-probabilities cheaply (only the last
position's logits are computed), which is what the calibrated judges need.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from evallab.providers.base import Completion, Request


class LlamaCppProvider:
    name = "llamacpp_local"

    def __init__(
        self, models: dict[str, str | Path], n_ctx: int = 4096, n_threads: int | None = None
    ) -> None:
        """``models`` maps a model alias (used in ``Request.model``) to a GGUF path."""
        self.paths = {k: Path(v) for k, v in models.items()}
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count() or 4
        self._loaded: dict = {}

    def _get(self, alias: str):
        if alias not in self._loaded:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            # keep one model resident at a time to bound memory
            self._loaded.clear()
            llm = Llama(str(self.paths[alias]), n_ctx=self.n_ctx, n_threads=self.n_threads, verbose=False)
            fmt = Jinja2ChatFormatter(llm.metadata["tokenizer.chat_template"], eos_token="", bos_token="")
            self._loaded[alias] = (llm, fmt)
        return self._loaded[alias]

    def _prompt(self, fmt, request: Request) -> str:
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        try:
            return fmt(messages=msgs).prompt
        except Exception:  # some templates (e.g. Gemma) reject a system role: fold it into user
            sys_txt = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
            rest = [m for m in msgs if m["role"] != "system"]
            if sys_txt and rest:
                rest[0] = {"role": "user", "content": sys_txt + "\n\n" + rest[0]["content"]}
            return fmt(messages=rest).prompt

    def complete(self, request: Request) -> Completion:
        import llama_cpp

        llm, fmt = self._get(request.model)
        prompt = self._prompt(fmt, request)
        tokens = llm.tokenize(prompt.encode(), add_bos=False, special=True)
        if len(tokens) + request.max_tokens > self.n_ctx:
            tokens = tokens[: self.n_ctx - request.max_tokens - 1]
        t0 = time.perf_counter()
        first: dict[str, float] = {}
        if request.top_logprobs:
            # Reuse the KV cache for the shared prefix (e.g. the same context checked against many
            # sentences): only the new suffix is evaluated.
            prev = list(llm.input_ids[: llm.n_tokens])
            k = 0
            for a, b in zip(prev, tokens[:-1], strict=False):
                if a != b:
                    break
                k += 1
            llm.n_tokens = k
            llm._ctx.kv_cache_seq_rm(-1, k, -1)
            llm.eval(tokens[k:])
            ptr = llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1)
            logits = np.ctypeslib.as_array(ptr, shape=(llm.n_vocab(),)).astype(np.float64)
            logp = logits - np.logaddexp.reduce(logits)
            for i in np.argsort(-logp)[: request.top_logprobs]:
                tok = llm.detokenize([int(i)]).decode("utf-8", errors="ignore")
                first[tok] = max(first.get(tok, -1e9), float(logp[i]))
            if request.max_tokens <= 1:
                text = max(first, key=lambda k: first[k])
                return Completion(text, len(tokens), 1, time.perf_counter() - t0, request.model, first)
        out = llm.create_completion(
            prompt=tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed if request.seed is not None else -1,
            stop=list(request.stop) or None,
        )
        latency = time.perf_counter() - t0
        usage = out["usage"]
        return Completion(
            text=out["choices"][0]["text"],
            prompt_tokens=int(usage["prompt_tokens"]),
            completion_tokens=int(usage["completion_tokens"]),
            latency_s=latency,
            model=request.model,
            first_token_logprobs=first,
        )
