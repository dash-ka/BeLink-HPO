#!/usr/bin/env python3
"""Run the `ontogpt` command with a fix for how it reads the model's reply.

Use it exactly like `ontogpt`:
    python ontogpt_fixed.py extract -t patient_phenotypes.yaml -i section.txt -m gpt-5.4 -o out.json

Why: OntoGPT 1.2.0 reads each reply line as "field: value". A field the model
leaves empty ("age_at_last_examination:") is taken as the start of a list
continued on the next line, and the next line is then thrown away. So every
time a patient's age (or sex) is empty, observed_spans, which comes right
after it, is lost. The same rule drops lists written one item per line, and
keeps only the first item of a bulleted list.

The fix tidies the reply before OntoGPT parses it:
  * an empty field is removed;
  * lines that continue a field (one item per line, bullets, numbering) are
    joined onto that field as one semicolon-separated line.
Everything else in OntoGPT is unchanged. JSON replies are left untouched.
"""
from __future__ import annotations

import json
import re
import sys

from ontogpt.engines import spires_engine as _se

FIELD = re.compile(r"^\s*[*#>\-\s]*([A-Za-z][A-Za-z_ ]*?)\s*\**\s*:\s*(.*)$")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def tidy_reply(text: str, slots: set[str]) -> str:
    """Put every field on one line and drop empty fields."""
    def is_slot(name: str) -> str | None:
        key = name.strip().lower().replace(" ", "_")
        if key in slots:
            return key
        if key.endswith("s") and key[:-1] in slots:
            return key[:-1]
        return None

    out: list[str] = []
    key, parts = None, []

    def flush():
        vals = [p.strip().rstrip(";").strip() for p in parts]
        vals = [v for v in vals if v]
        if key and vals:
            out.append(f"{key}: " + "; ".join(vals))

    for line in text.splitlines():
        m = FIELD.match(line)
        k = is_slot(m.group(1)) if m else None
        if k:
            flush()
            key, parts = k, [m.group(2)]
        elif key and line.strip() and not line.strip().startswith("`" * 3):
            parts.append(BULLET.sub("", line))
        elif not key:
            out.append(line)
    flush()
    return "\n".join(out)


_original = _se.SPIRESEngine._parse_response_to_dict


def _parse_response_to_dict(self, results, cls=None):
    try:
        json.loads(results)
        return _original(self, results, cls)            # JSON replies are parsed as before
    except (ValueError, TypeError):
        pass
    try:
        name = self._require_template_class(cls).name
        slots = {s.lower() for s in self.schemaview.class_slots(name)}
    except Exception:                                   # cannot tell the fields: leave it alone
        return _original(self, results, cls)
    return _original(self, tidy_reply(results, slots), cls)


_se.SPIRESEngine._parse_response_to_dict = _parse_response_to_dict

if __name__ == "__main__":
    from ontogpt.cli import main
    sys.argv[0] = "ontogpt"
    sys.exit(main())
