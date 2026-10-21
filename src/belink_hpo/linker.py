"""Query expansion + retrieval + re-ranking in one call."""

from __future__ import annotations


class HPOLinker:
    """Link mentions to HPO in up to three steps:

    1. expansion (optional): an LLM generates a standard HPO name for the mention;
    2. retrieval: the bi-encoder finds candidates, with the standard name mixed into the query;
    3. re-ranking: the fine-tuned model picks one candidate, or "None of the above".

        linker = HPOLinker(
            CandidateRetriever.from_files("index/hpo_vec.npy", "index/ontology_mapping.txt"),
            HPOReranker.from_pretrained(preferred_names=load_preferred_names("index/preferred_names.json")),
            expander=StandardNameGenerator(),   # leave out to skip expansion
        )
        linker.link("Hands show marked brachydactyly.", "brachydactyly").id

    ``expander`` is any callable ``(mention, context) -> name or None``. A standard name
    passed in explicitly is used as is and never regenerated.
    """

    def __init__(self, retriever, reranker, expander=None, top_k=30, rocchio=True, alpha=0.6):
        self.retriever = retriever
        self.reranker = reranker
        self.expander = expander
        self.top_k = top_k
        self.rocchio = rocchio
        self.alpha = alpha

    def expand(self, items):
        """Standard names for the items: the given one, else the expander's, else None."""
        names = []
        for it in items:
            name = it.get("standard_name")
            if not name and self.expander is not None:
                name = self.expander(it["mention"], it["context"])
            names.append(name or None)
        return names

    def link_batch(self, items, batch_size=32, return_standard_names=False):
        """``items`` are dicts with "context", "mention" and optionally "standard_name"."""
        standard_names = self.expand(items)
        candidates = self.retriever.retrieve(
            [it["mention"] for it in items],
            standard_names if any(standard_names) else None,
            self.top_k, self.rocchio, self.alpha,
        )
        results = self.reranker.rerank_batch(
            [{"context": it["context"], "mention": it["mention"], "candidates": c} for it, c in zip(items, candidates)],
            batch_size=batch_size,
        )
        return (results, standard_names) if return_standard_names else results

    def link(self, context, mention, standard_name=None):
        return self.link_batch([{"context": context, "mention": mention, "standard_name": standard_name}])[0]
