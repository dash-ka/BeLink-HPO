"""Retrieval + re-ranking in one call."""

from __future__ import annotations


class HPOLinker:
    """Link mentions to HPO: retrieve candidates with the bi-encoder, then let the re-ranker choose.

        linker = HPOLinker(
            CandidateRetriever.from_files("index/hpo_vec.npy", "index/ontology_mapping.txt"),
            HPOReranker.from_pretrained(preferred_names=load_preferred_names("hp.obo")),
        )
        linker.link("Hands show marked brachydactyly.", "brachydactyly").id
    """

    def __init__(self, retriever, reranker, top_k=30, rocchio=True, alpha=0.6):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rocchio = rocchio
        self.alpha = alpha

    def link_batch(self, items, batch_size=32):
        """``items`` are dicts with "context", "mention" and optionally "standard_name"."""
        standard_names = [it.get("standard_name") for it in items]
        candidates = self.retriever.retrieve(
            [it["mention"] for it in items],
            standard_names if any(standard_names) else None,
            self.top_k, self.rocchio, self.alpha,
        )
        return self.reranker.rerank_batch(
            [{"context": it["context"], "mention": it["mention"], "candidates": c} for it, c in zip(items, candidates)],
            batch_size=batch_size,
        )

    def link(self, context, mention, standard_name=None):
        return self.link_batch([{"context": context, "mention": mention, "standard_name": standard_name}])[0]
