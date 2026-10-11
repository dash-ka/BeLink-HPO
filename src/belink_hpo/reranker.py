"""Load the fine-tuned adapter and pick the right HPO term among retrieved candidates.

The model is not asked to generate text. Each prompt gets one forward pass, and the
scores of the option letters (A, B, C, ...) at the answer position are compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from . import DEFAULT_MODEL
from .prompt import LETTERS, MAX_OPTIONS, build_user_message, select_options


@dataclass
class RerankResult:
    id: str | None          # chosen HPO id, or None when the model answers "None of the above."
    name: str | None        # option text as shown to the model
    letter: str
    confidence: float       # probability of the chosen letter among the letters offered
    options: list = field(default_factory=list)  # [(letter, id, name), ...] as shown to the model


def load_model(adapter=DEFAULT_MODEL, load_in_4bit=False, device_map="auto", token=None):
    """Load base model + LoRA adapter. Returns (model, tokenizer).

    Without 4-bit loading the adapter is merged into the base weights, which is the
    fastest option. With 4-bit loading the adapter stays separate (less VRAM, slower).
    """
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = dict(device_map=device_map, token=token)
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(adapter, dtype=dtype, **kwargs)
    except TypeError:  # older transformers versions call this argument torch_dtype
        model = AutoPeftModelForCausalLM.from_pretrained(adapter, torch_dtype=dtype, **kwargs)

    base_name = model.peft_config["default"].base_model_name_or_path
    if load_in_4bit:
        model = model.get_base_model()    # adapter layers stay active inside it
    else:
        model = model.merge_and_unload()  # fold the adapter into the weights
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_name, token=token)
    return model, tokenizer


class HPOReranker:
    """Multiple-choice re-ranker over HPO candidates.

        reranker = HPOReranker.from_pretrained()
        result = reranker.rerank(
            context="A syndrome of brachydactyly is reported in five members of a family.",
            mention="brachydactyly",
            candidates=[{"id": "HP:0001831", "name": "short toe"}, {"id": "HP:0001156", "name": "brachydactyly"}],
        )
        result.id  # "HP:0001156"
    """

    def __init__(self, model, tokenizer, preferred_names=None, max_options=MAX_OPTIONS):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"  # last position of every row = end of the prompt
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.preferred_names = preferred_names
        self.max_options = max_options

    @classmethod
    def from_pretrained(cls, adapter=DEFAULT_MODEL, load_in_4bit=False, device_map="auto", token=None,
                        preferred_names=None, max_options=MAX_OPTIONS):
        """``adapter`` is a Hub id or a local adapter directory (or a checkpoint inside one).

        ``preferred_names`` maps concept id -> preferred name. When given, each candidate
        is shown as "name (aka preferred name)", as in the training data.
        """
        model, tokenizer = load_model(adapter, load_in_4bit, device_map, token)
        return cls(model, tokenizer, preferred_names, max_options)

    def _label_token_ids(self, labels):
        ids = []
        for label in labels:
            toks = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(toks) != 1:
                raise ValueError(f"Label {label!r} is not a single token for this tokenizer.")
            ids.append(toks[0])
        return ids

    @torch.inference_mode()
    def score(self, user_messages, labels=LETTERS, batch_size=32, progress=False):
        """Return a [len(user_messages), len(labels)] tensor of label logits, in input order."""
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # must match training
            )
            for text in user_messages
        ]
        encoded = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]

        # Longest first: similar lengths share a batch (little padding), and an
        # out-of-memory error shows up on the first batch instead of mid-run.
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]), reverse=True)

        model = self.model
        backbone = getattr(model, model.base_model_prefix)  # transformer without the output layer
        head = model.get_output_embeddings().weight         # [vocab, hidden]
        label_ids = torch.tensor(self._label_token_ids(labels), device=head.device)
        label_head = head[label_ids].float()                # [num_labels, hidden]

        logits = torch.empty(len(encoded), len(labels))
        batches = range(0, len(order), batch_size)
        for start in tqdm(batches, desc="Re-ranking") if progress else batches:
            idx = order[start : start + batch_size]
            batch = self.tokenizer.pad({"input_ids": [encoded[i] for i in idx]}, return_tensors="pt")
            batch = {k: v.to(model.device) for k, v in batch.items()}

            hidden = backbone(**batch).last_hidden_state[:, -1]  # answer position only
            # Score only the option letters instead of the full vocabulary
            logits[idx] = (hidden.to(label_head.device).float() @ label_head.T).cpu()
        return logits

    def predict_labels(self, user_messages, labels=LETTERS, batch_size=32, progress=False):
        """Return (predicted_label, confidence) for ready-made user messages."""
        if not user_messages:
            return []
        probs = self.score(user_messages, labels, batch_size, progress).softmax(dim=-1)
        conf, best = probs.max(dim=-1)
        return [(labels[b], c) for b, c in zip(best.tolist(), conf.tolist())]

    def rerank_batch(self, items, batch_size=32, progress=False):
        """Re-rank many mentions. ``items`` are dicts with "context", "mention" and "candidates".

        Candidates are dicts with "id" (or "cui") and "name", or (id, name) pairs, in
        retrieval order. Returns one RerankResult per item.
        """
        option_sets = [select_options(it["candidates"], self.preferred_names, self.max_options) for it in items]
        messages = [
            build_user_message(it["context"], it["mention"], [name for _, name in opts])
            for it, opts in zip(items, option_sets)
        ]
        if not messages:
            return []
        logits = self.score(messages, LETTERS, batch_size, progress)

        results = []
        for row, opts in zip(logits, option_sets):
            n = len(opts) + 1                      # the options plus "None of the above."
            probs = row[:n].softmax(dim=-1)        # letters that were not offered cannot win
            best = int(probs.argmax())
            cid, name = opts[best] if best < len(opts) else (None, None)
            results.append(RerankResult(
                id=cid,
                name=name,
                letter=LETTERS[best],
                confidence=float(probs[best]),
                options=[(LETTERS[i], oid, oname) for i, (oid, oname) in enumerate(opts)],
            ))
        return results

    def rerank(self, context, mention, candidates):
        """Pick the HPO term for one mention. See ``rerank_batch``."""
        return self.rerank_batch([{"context": context, "mention": mention, "candidates": candidates}])[0]
