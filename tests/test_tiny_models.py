"""End-to-end checks with tiny randomly initialised models built on the fly (no downloads, CPU only).

They exercise the real code paths: LoRA training, saving and re-loading the adapter,
letter scoring, index building and candidate retrieval.
"""

import gzip
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("datasets")
pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from belink_hpo.dataset import load_jsonl, prepare_test, save_jsonl  # noqa: E402
from belink_hpo.prompt import LETTERS  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)

EXAMPLES = [
    {"doc_id": f"d{i}", "context": f"Patient {i} shows marked brachydactyly of both hands.", "mention": "brachydactyly",
     "gold_id": "HP:1", "candidates": [{"id": "HP:2", "name": "short toe"}, {"id": "HP:1", "name": "brachydactyly"},
                                       {"id": "HP:3", "name": "oligodactyly"}]}
    for i in range(24)
]


SPECIALS = ["<pad>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def _save_qwen_style_tokenizer(texts, path):
    """Train a small byte-level BPE tokenizer (the kind Qwen uses) and return it as reloaded from disk."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=400, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<|im_end|>", additional_special_tokens=SPECIALS[1:])
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)
    return transformers.AutoTokenizer.from_pretrained(path)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """A tiny Qwen-style base model, a dataset, and an adapter trained on it with `belink-hpo train`."""
    from belink_hpo import train

    root = tmp_path_factory.mktemp("belink")
    # Fixed option order, so the answer is always "B": something a tiny random model can learn,
    # which lets the tests check that training, saving, loading and scoring agree with each other.
    rows = prepare_test(EXAMPLES[:20]) * 3
    test_rows = prepare_test(EXAMPLES[20:])
    save_jsonl(rows, root / "data" / "train.jsonl")
    save_jsonl(test_rows, root / "data" / "test.jsonl")

    texts = [m["content"] for r in rows + test_rows for m in r["messages"]] + ["user assistant"]
    config = transformers.Qwen2Config(
        vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
    )
    config.save_pretrained(root / "base")  # written first so the tokenizer reloads as a Qwen tokenizer
    tokenizer = _save_qwen_style_tokenizer(texts, root / "base")
    assert len(tokenizer) <= config.vocab_size
    config.pad_token_id, config.eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    torch.manual_seed(0)
    transformers.AutoModelForCausalLM.from_config(config).save_pretrained(root / "base")

    train.main([
        "--model", str(root / "base"), "--data-dir", str(root / "data"), "--output-dir", str(root / "adapter"),
        "--no-4bit", "--epochs", "8", "--lr", "2e-2", "--batch-size", "4", "--grad-accum", "1", "--eval-steps", "50",
    ])
    return root


def test_training_saves_an_adapter(workdir):
    assert (workdir / "adapter" / "adapter_config.json").exists()
    assert any((workdir / "adapter").glob("adapter_model.*"))


def test_scores_match_a_full_forward_pass(workdir):
    from belink_hpo.reranker import HPOReranker

    reranker = HPOReranker.from_pretrained(str(workdir / "adapter"))
    users = [r["messages"][0]["content"] for r in load_jsonl(workdir / "data" / "test.jsonl")]
    users.append(users[0].replace("Patient", "Patient Patient Patient"))  # a longer row, to force padding
    scores = reranker.score(users, LETTERS, batch_size=3)
    assert scores.shape == (len(users), 26)

    tok, model = reranker.tokenizer, reranker.model
    label_ids = [tok(letter, add_special_tokens=False)["input_ids"][0] for letter in LETTERS]
    for text, row in zip(users, scores):
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        assert prompt.endswith("<think>\n\n</think>\n\n")
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            full = model(**ids).logits[0, -1, label_ids].float()
        assert torch.allclose(row, full, atol=1e-4)


def test_rerank_learned_the_task_and_maps_letters_to_ids(workdir):
    from belink_hpo.reranker import HPOReranker

    reranker = HPOReranker.from_pretrained(str(workdir / "adapter"))
    results = reranker.rerank_batch(EXAMPLES[20:])
    for res in results:
        assert [o[0] for o in res.options] == ["A", "B", "C"]
        assert res.letter in "ABCD" and 0.0 <= res.confidence <= 1.0
        assert res.id == {o[0]: o[1] for o in res.options}.get(res.letter)  # None for "None of the above."
    assert all(r.letter == "B" and r.id == "HP:1" for r in results)

    single = reranker.rerank(**{k: EXAMPLES[20][k] for k in ("context", "mention", "candidates")})
    assert single.letter == results[0].letter


def test_predict_command(workdir, capsys):
    from belink_hpo import cli

    out = workdir / "predictions.jsonl"
    cli.main(["predict", "--adapter", str(workdir / "adapter"), "--file", str(workdir / "data" / "test.jsonl"),
              "--out", str(out), "--batch-size", "2"])
    preds = load_jsonl(out)
    assert len(preds) == 4 and all(p["gold"] == "B" and p["pred"] == "B" for p in preds)
    assert "Accuracy: 100.00% on 4 labelled rows" in capsys.readouterr().out


NAMES = [("hp:1", "Brachydactyly"), ("hp:1", "short fingers"), ("hp:2", "short toe"), ("hp:3", "oligodactyly")]


def _build_index(tmp_path):
    """A tiny random BERT encoder and an index of NAMES built with `belink-hpo embed-terms`."""
    from belink_hpo import cli

    with gzip.open(tmp_path / "terms.json.gz", "wt", encoding="utf-8") as f:
        json.dump([{"id": i, "name": n} for i, n in NAMES], f)

    (tmp_path / "encoder").mkdir()
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted({w for _, n in NAMES for w in n.lower().split()})
    (tmp_path / "encoder" / "vocab.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizer(str(tmp_path / "encoder" / "vocab.txt"))
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
    )).save_pretrained(tmp_path / "encoder")
    tokenizer.save_pretrained(tmp_path / "encoder")

    vec = tmp_path / "index" / "vec.npy"
    cli.main(["embed-terms", "--ontology", str(tmp_path / "terms.json.gz"), "--model_name", str(tmp_path / "encoder"),
              "--out_vectors", str(vec)])
    return vec, tmp_path / "index" / "ontology_mapping.txt", tmp_path / "encoder"


def test_index_building_and_retrieval(tmp_path):
    from belink_hpo import cli
    from belink_hpo.retrieval import CandidateRetriever, parse_candidates

    vec, mapping, _ = _build_index(tmp_path)
    retriever = CandidateRetriever.from_files(vec, mapping, str(tmp_path / "encoder"))
    (plain,) = retriever.retrieve(["short toe"], top_k=2)
    assert plain[0] == {"id": "HP:2", "name": "short toe"}  # an indexed name retrieves itself first
    # feedback: alpha=0 means the query is the standard name alone
    (fed,) = retriever.retrieve(["short toe"], standard_names=["oligodactyly"], top_k=1, alpha=0.0)
    assert fed[0]["id"] == "HP:3"
    (unfed,) = retriever.retrieve(["short toe"], standard_names=[None], top_k=1)
    assert unfed[0]["id"] == "HP:2"

    records = {"a": {"label": "short toe", "standard_name": "oligodactyly"}, "b": {"label": "oligodactyly"}}
    (tmp_path / "mentions.json").write_text(json.dumps(records), encoding="utf-8")
    cli.main(["retrieve", "--input", str(tmp_path / "mentions.json"), "--output_file", str(tmp_path / "out.json"),
              "--kb_vectors", str(vec), "--model_name", str(tmp_path / "encoder"), "--top_k", "2"])
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert out["a"]["candidates"].startswith("hp:2||short toe|")
    assert parse_candidates(out["b"]["candidates"])[0] == {"id": "hp:3", "name": "oligodactyly"}


class FakeLLM:
    """Stands in for the OpenAI client: answers every prompt with the same standard name."""

    def __init__(self, name):
        self.name, self.prompts = name, []
        self.chat = self.completions = self

    def create(self, model, messages, **kwargs):
        from types import SimpleNamespace

        self.prompts.append(messages[0]["content"])
        content = json.dumps({"name": self.name})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_linking_with_query_expansion(workdir, tmp_path, monkeypatch, capsys):
    from belink_hpo import cli
    from belink_hpo.expansion import StandardNameGenerator
    from belink_hpo.linker import HPOLinker
    from belink_hpo.reranker import HPOReranker
    from belink_hpo.retrieval import CandidateRetriever

    vec, mapping, encoder = _build_index(tmp_path)
    llm = FakeLLM("oligodactyly")
    # alpha=0: the query is the standard name alone, so the expansion decides which concept comes first
    linker = HPOLinker(CandidateRetriever.from_files(vec, mapping, str(encoder)),
                       HPOReranker.from_pretrained(str(workdir / "adapter")),
                       expander=StandardNameGenerator(client=llm), top_k=4, alpha=0.0)

    results, names = linker.link_batch([{"context": "Feet with short toe.", "mention": "short toe"}],
                                       return_standard_names=True)
    assert names == ["oligodactyly"]
    assert results[0].options[0][1] == "HP:3"
    assert "short toe" in llm.prompts[0] and "Feet with short toe." in llm.prompts[0]

    # an explicit standard name is used as is, without calling the LLM
    explicit = linker.link("Feet with short toe.", "short toe", standard_name="short toe")
    assert explicit.options[0][1] == "HP:2" and len(llm.prompts) == 1

    # the same through `belink-hpo link --expand`, with the OpenAI client replaced by the fake
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: FakeLLM("oligodactyly"))
    (tmp_path / "mentions.jsonl").write_text(
        json.dumps({"context": "Feet with short toe.", "mention": "short toe", "gold_id": "HP:3"}) + "\n",
        encoding="utf-8")
    cli.main(["link", "--input", str(tmp_path / "mentions.jsonl"), "--out", str(tmp_path / "linked.jsonl"),
              "--kb-vectors", str(vec), "--encoder", str(encoder), "--adapter", str(workdir / "adapter"),
              "--expand", "--alpha", "0", "--top-k", "4"])
    (row,) = load_jsonl(tmp_path / "linked.jsonl")
    assert row["standard_name"] == "oligodactyly" and row["mention"] == "short toe"
    assert set(row) == {"context", "mention", "standard_name", "id", "name", "confidence"}
    assert "Accuracy:" in capsys.readouterr().out
