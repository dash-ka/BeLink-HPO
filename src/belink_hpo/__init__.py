"""BeLink-HPO: an LLM re-ranker for linking phenotype mentions to HPO terms."""

__version__ = "0.1.0"

DEFAULT_MODEL = "Dash00/qwen3-4b-lora-hpo-reranker"

_LAZY = {
    "HPOReranker": "belink_hpo.reranker",
    "RerankResult": "belink_hpo.reranker",
    "CandidateRetriever": "belink_hpo.retrieval",
    "HPOLinker": "belink_hpo.linker",
    "OntoReader": "belink_hpo.ontology",
}

__all__ = ["DEFAULT_MODEL", *_LAZY]


def __getattr__(name):
    # Imported on first use so that the light modules (prompt, dataset) work without loading torch.
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'belink_hpo' has no attribute {name!r}")
