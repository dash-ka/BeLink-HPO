from belink_hpo.dataset import accuracy_at_1, prepare_test, prepare_train, split_examples
from belink_hpo.prompt import augment_name, build_user_message, select_options

PREF = {"HP:1": "Brachydactyly", "HP:2": "Short toe", "HP:3": "Oligodactyly"}
CANDS = [
    {"cui": "HP:2", "name": "short toe (brachydactyly of the foot)"},
    {"id": "HP:1", "name": "short fingers or toes (brachydactyly)"},
    {"id": "HP:1", "name": "brachydactyly"},
    ("HP:3", "oligodactyly"),
]
EXAMPLE = {"doc_id": "d1", "context": "Marked brachydactyly.", "mention": "brachydactyly", "gold_id": "HP:1", "candidates": CANDS}


def test_augment_name():
    assert augment_name("short toe", "Brachydactyly") == "short toe (aka brachydactyly)"
    assert augment_name("short toe (brachydactyly)", "Brachydactyly") == "short toe (aka brachydactyly)"
    assert augment_name("brachydactyly", "Brachydactyly") == "brachydactyly"
    assert augment_name("short toe", None) == "short toe"


def test_select_options_keeps_first_name_per_concept_in_order():
    assert select_options(CANDS, PREF) == [
        ("HP:2", "short toe (brachydactyly of the foot) (aka short toe)"),
        ("HP:1", "short fingers or toes (aka brachydactyly)"),
        ("HP:3", "oligodactyly"),
    ]
    assert len(select_options(CANDS, PREF, max_options=2)) == 2


def test_user_message_format():
    msg = build_user_message("Marked brachydactyly.", "brachydactyly", ["short toe", "brachydactyly"])
    assert msg == (
        "<Instruct>: Given the context 'Marked brachydactyly.', select the correct biomedical concept "
        "corresponding to 'brachydactyly'. Answer using one of the provided options.\n\n"
        "<Options>: A: short toe\nB: brachydactyly\nC: None of the above."
    )


def test_prepare_test_row():
    (row,) = prepare_test([EXAMPLE], PREF)
    assert row["messages"][1] == {"role": "assistant", "content": "B"}
    assert row["messages"][0]["content"].endswith("D: None of the above.")


def test_prepare_test_none_of_the_above_and_skips():
    missing_gold = dict(EXAMPLE, gold_id="HP:9")
    no_candidates = dict(EXAMPLE, candidates=[])
    rows = prepare_test([missing_gold, no_candidates], PREF)
    assert len(rows) == 1 and rows[0]["messages"][1]["content"] == "D"
    assert len(prepare_test([EXAMPLE, EXAMPLE], PREF, unique_mentions=True)) == 1


def _answer_text(row):
    user, letter = row["messages"][0]["content"], row["messages"][1]["content"]
    options = user.split("<Options>: ", 1)[1].split("\n")
    return next(line.split(": ", 1)[1] for line in options if line.startswith(f"{letter}: "))


def test_prepare_train_answer_points_at_gold():
    rows = prepare_train([EXAMPLE], PREF, epochs=20)
    assert len(rows) == 20
    for row in rows:
        assert _answer_text(row) in {"short fingers or toes (aka brachydactyly)", "brachydactyly"}
        assert row["messages"][0]["content"].count("\n") == 5  # blank line + 3 options + none


def test_prepare_train_nota():
    rows = prepare_train([EXAMPLE], PREF, epochs=10, nota_rate=1.0)
    assert all(_answer_text(r) == "None of the above." for r in rows)
    rows = prepare_train([dict(EXAMPLE, gold_id="HP:9")], PREF, epochs=3)
    assert all(_answer_text(r) == "None of the above." for r in rows)


def test_split_and_accuracy():
    examples = [dict(EXAMPLE, doc_id=f"d{i // 2}") for i in range(20)]
    train, test = split_examples(examples, test_fraction=0.2)
    assert len(train) == 16 and len(test) == 4
    assert not {e["doc_id"] for e in train} & {e["doc_id"] for e in test}
    assert accuracy_at_1([EXAMPLE]) == 0.0
    assert accuracy_at_1([dict(EXAMPLE, gold_id="HP:2")]) == 1.0


def test_linker_passes_retrieved_candidates_to_the_reranker():
    from belink_hpo.linker import HPOLinker

    calls = {}

    class Retriever:
        def retrieve(self, mentions, standard_names, top_k, rocchio, alpha):
            calls["retrieve"] = (mentions, standard_names, top_k, rocchio, alpha)
            return [[{"id": "HP:1", "name": m}] for m in mentions]

    class Reranker:
        def rerank_batch(self, items, batch_size=32):
            calls["rerank"] = items
            return [it["candidates"][0]["id"] for it in items]

    linker = HPOLinker(Retriever(), Reranker())
    assert linker.link("ctx", "short toe") == "HP:1"
    assert calls["retrieve"] == (["short toe"], None, 30, True, 0.6)
    assert calls["rerank"] == [{"context": "ctx", "mention": "short toe", "candidates": [{"id": "HP:1", "name": "short toe"}]}]
    linker.link("ctx", "short toe", standard_name="Short toe")
    assert calls["retrieve"][1] == ["Short toe"]


def test_prepare_obo_command(tmp_path):
    import gzip
    import json

    from belink_hpo import cli

    (tmp_path / "hp.obo").write_text(
        "format-version: 1.2\n\n"
        "[Term]\nid: HP:0000001\nname: All\n\n"
        "[Term]\nid: HP:0001831\nname: Short toe\nsynonym: \"Stubby toes\" EXACT layperson []\n\n"
        "[Term]\nid: HP:0000002\nname: Old thing\nis_obsolete: true\n",
        encoding="utf-8",
    )
    cli.main(["prepare-obo", "--ontology_path", str(tmp_path / "hp.obo"), "--output_dir", str(tmp_path / "index")])
    with gzip.open(tmp_path / "index" / "terminology.json.gz", "rt", encoding="utf-8") as f:
        names = json.load(f)
    assert names == [
        {"id": "hp:0001831", "name": "short toe (stubby toes)"},
        {"id": "hp:0001831", "name": "stubby toes (short toe)"},
    ]
    preferred = json.loads((tmp_path / "index" / "preferred_names.json").read_text(encoding="utf-8"))
    assert preferred == {"HP:0001831": "Short toe"}
