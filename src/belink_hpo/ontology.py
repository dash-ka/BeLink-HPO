"""Read HPO term names and alternate ids from hp.obo. Requires: pip install "belink-hpo[data]"."""

from __future__ import annotations

import json


class OntoReader:
    """Primary ids, preferred names and alternate ids of an OBO ontology.

        reader = OntoReader("hp.obo")
        reader.consolidate("HP:0009381")   # primary id, or None if unknown
        reader.terms["HP:0001156"]         # "Brachydactyly"
    """

    def __init__(self, onto_file):
        from pronto import Ontology

        self.terms = {}             # primary id -> preferred name
        self.reverse_alt_ids = {}   # alternate id -> primary id
        for term in Ontology(onto_file).terms():
            self.terms[term.id] = term.name
            for alt_id in term.alternate_ids or ():
                self.reverse_alt_ids[alt_id] = term.id

    def consolidate(self, hpo_id):
        """Map an id to its primary id; None if the ontology does not know it."""
        if hpo_id in self.terms:
            return hpo_id
        return self.reverse_alt_ids.get(hpo_id)

    def normalize(self, hpo_id):
        """'hp:0001156' -> 'HP:0001156', with alternate ids mapped to the primary id."""
        hpo_id = hpo_id.strip().upper()
        return self.consolidate(hpo_id) or hpo_id

    def save_preferred_names(self, path):
        """Write {id: preferred name} as JSON, for use without pronto or hp.obo."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.terms, f, ensure_ascii=False)


def load_preferred_names(path):
    """Load {id: preferred name} from a JSON file, or straight from an .obo file."""
    if str(path).endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return OntoReader(path).terms
