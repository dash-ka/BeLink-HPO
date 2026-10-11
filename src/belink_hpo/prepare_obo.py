#!/usr/bin/env python
"""Turn an OBO ontology into the list of names that the retrieval index is built from.

    belink-hpo prepare-obo --ontology_path hp.obo --output_dir index

Writes to the output directory:
    terminology.json.gz    [{"id": "hp:0001156", "name": "short fingers or toes (brachydactyly)"}, ...]
                           one record per name; this is the input of `belink-hpo embed-terms`
    processed_obo.json.gz  the parsed terms (name, definition, synonyms, parents, alternate ids)
    alt_ids2cui.json       alternate id -> primary id
    statistics_kb.json     counts

Every synonym is written as "synonym (preferred name)" and every preferred name as
"preferred name (longest synonym)", so that no two concepts share a name. Obsolete terms
and the root term "All" are skipped. The file name must be the ontology prefix (hp.obo):
terms whose id does not contain it are treated as external and skipped. If the file does
not exist it is downloaded from purl.obolibrary.org.
"""

import argparse, re, json, gzip
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def manual_parse_obo(file_path):
    "Reads OBO file and outputs processed dictionary."

    count_missing_name = 0
    count_obsolete_terms = 0
    count_external_terms = 0
    
    # Regex Patterns

    # Matches: id: ID
    id_pattern = re.compile(r'^id:\s+(\S+)', re.MULTILINE)
    # Matches: alt_id: ID
    alt_id_pattern = re.compile(r'^alt_id:\s+(\S+)', re.MULTILINE)
    # Matches: name: NAME
    name_pattern = re.compile(r'^name:\s+(.*)', re.MULTILINE)
    # Matches: def: "TEXT" [refs]
    def_pattern = re.compile(r'^def:\s+"([^"]+)"\s*(.*)', re.MULTILINE)
    # Matches: is_a: ID
    isa_pattern = re.compile(r'^is_a:\s+(\S+)', re.MULTILINE)
    # Matches: is_obsolete: true
    obs_pattern = re.compile(r'^is_obsolete:\s+true', re.MULTILINE)
    # Matches: synonym: "text" SCOPE TYPE [refs]
    syn_pattern = re.compile(r'^synonym:\s+"(?P<text>[^"]+)"\s+(?P<scope>\w+)\s+(?P<type>[^\[\s]+)?\s*\[(?P<refs>[^\]]*)\]', re.MULTILINE)

    id_to_name = {}
    
    # --- Pass 1: Quick Scan for ID -> Name mapping ---
    path = Path(file_path)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        term_blocks = content.split('[Term]')

    for block in term_blocks:
        id_match = id_pattern.search(block)
        name_match = name_pattern.search(block)
        if id_match and name_match:
            t_id = id_match.group(1).strip().lower()
            t_name = name_match.group(1).split('!')[0].strip()
            id_to_name[t_id] = t_name

    # --- Pass 2: Extract Full Structured Data ---
    #     
    
    ontology_data = {}
    OBO_PREFIX = path.stem.lower() 

    for block in tqdm(term_blocks, desc="Parsing OBO terms..."):
        if not block.strip():
            continue
            
        id_match = id_pattern.search(block)
        if not id_match:
            continue
        
        term_id = id_match.group(1).strip().lower()
        if OBO_PREFIX not in term_id:
            count_external_terms += 1
            continue

        # skipping obsolete terms (Defaults to False if tag is missing)
        if obs_pattern.search(block):
            count_obsolete_terms +=1
            continue
        
        # 1. Basic Metadata
        name_match = name_pattern.search(block)
        name = name_match.group(1).split('!')[0].strip() if name_match else ""
        if not name:
            count_missing_name +=1
            continue

        # 2. Definition
        def_match = def_pattern.search(block)
        definition_text = def_match.group(1) if def_match else ""

        # 3. Detailed Synonyms
        synonyms = []
        for m in syn_pattern.finditer(block):
            synonym_text = m.group("text").strip()

            # --- APPLY FILTER LOGIC ---
            junk_chars = {"[*-]", "."}
            inchi_keys = {"inchikey", "inchi"}

            # 1. Skip if contains inchi/inchikey (case-insensitive)
            if any(x in synonym_text.lower() for x in inchi_keys):
                continue
            
            # 2. Skip if it matches specific junk characters
            if synonym_text in junk_chars:
                continue
            
            # 3. If it passes, add the structured data
            #synonyms.append({
            #    "text": synonym_text,
            #    "scope": m.group("scope"),
            #    "type": m.group("type").strip() if m.group("type") else None,
            #    "xrefs": [r.strip() for r in m.group("refs").split(",")] if m.group("refs") else []
            #})
            synonyms.append(synonym_text)

        # 4. Parents (is_a)
        is_a_ids = isa_pattern.findall(block)
        parents = [{"id": pid.lower(), "name": id_to_name.get(pid.lower(), "Unknown Name")} for pid in is_a_ids]

        # 5. Alternative ids
        alt_ids = [aid.group(1).strip().lower() for aid in alt_id_pattern.finditer(block)]

        # 6. Final Dictionary Assembly
        ontology_data[term_id] = {
                "cui": term_id,
                "name": name,
                "definition": definition_text,
                "synonyms": "|".join(synonyms),
                "is_a": json.dumps(parents),
                "alt_cuis": "|".join(alt_ids)
            }

    print(f"Done! Parsed {len(ontology_data)} terms.")
    print("   # missing names: ", count_missing_name)
    print("   # obsolete terms: ", count_obsolete_terms)
    print("   # external terms: ", count_external_terms)
    return ontology_data


def load_obo(path_to_obo_file, cache_obo=True):
    """
    Loads an OBO ontology. If the file doesn't exist locally, 
    attempts to download it from PURL based on the filename.
    """

    path = Path(path_to_obo_file)
    
    if not path.exists():

        obo_prefix = path.stem 
        url = f"http://purl.obolibrary.org/obo/{obo_prefix}.obo"
    
        print(f"File not found locally. Fetching from URL: {url}")
        
        try:
            import requests
        except ImportError:
            raise SystemExit(f"{path} not found. Download it, or run 'pip install requests' to fetch it automatically.")

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        path.parent.mkdir(parents=True, exist_ok=True)
        
        if cache_obo:
            with open(path, 'wb') as f:
                f.write(response.content)
        print("Download complete.")

    print(f"Processing {str(path)} file.")
    return manual_parse_obo(path)


def get_name2cuis_mapping(kb: dict) -> tuple:

    alt_ids2cui, new_kb = dict(), dict()
    name2cui = defaultdict(list)
    pattern = re.compile("[+;]")

    for cui, entry in tqdm(kb.items(), total=len(kb), desc="Processing KB"):
        if cui == "1":
            continue

        alt_cuis = entry["alt_cuis"].split("|")

        # Normalize CUI by removing composite identifiers
        pattern = re.compile("[+;,]")
        cui = pattern.sub("|", cui)
        if "|" in cui:
            composite = [c.strip() for c in cui.split("|") if c.strip()]
            cui, extras = composite[0], composite[1:]
            alt_cuis.extend(extras)
            print("Composite CUI resolved:", composite)

        # Map alternative IDs
        for alt in alt_cuis:
            if alt != cui:
                alt_ids2cui[alt] = cui

        pref_name = entry["name"].lower()
        if not pref_name:
            continue

        name2cui[pref_name].append(cui)
        synonyms = []
        for syn in entry["synonyms"].split("|"):
            syn = syn.lower().strip()
            if not syn or syn == pref_name:
                continue
            if len(syn) < 5:
                syn = f"{syn} ({pref_name})"
            
            name2cui[syn].append(cui)
            synonyms.append(syn)

        new_kb[cui] = {"name":pref_name, "synonyms":synonyms}

    print(f"Old KB #entities: {len(kb)}\nNew KB # entities: {len(new_kb)}" )
    return name2cui, new_kb, alt_ids2cui


def resolve_homonyms(name2cui: list, kb: dict, alt_ids2cui: dict) -> dict:
    
    """Resolve duplicate term names mapping to different CUIs."""
    
    homonyms, failed = 0, 0
    avg_names = []
    is_homonym = lambda x: len(set(name2cui.get(x, []))) > 1
    disambiguated_aliases = dict()

    for cui, entity in tqdm(kb.items(), desc="Resolving homonyms"):
        
        cui = alt_ids2cui.get(cui, cui)
        pref_name = entity["name"]
        synonyms = entity.get("synonyms")
        synonyms = sorted(set(synonyms) if synonyms else [], key=len, reverse=True)
        
        avg_names.append(len([pref_name] + synonyms))
        for name in [pref_name] + synonyms:
            # find homonyms (same name, different CUI)
            if is_homonym(name): 
                
                homonyms += 1
            # If the preferred name is different, disambiguate
            if name != pref_name:
                new_name = f"{name} ({pref_name})"

                # If the "name + preferred name" exists, disambiguate using synonyms
                if is_homonym(new_name):
                    for s in synonyms:
                            if s not in [name, pref_name]:
                                new_name = f"{name} ({s})"
                                break
                name = new_name
            
            # If preferred name is the same, disambiguate with a synonym
            else:
                for s in synonyms:
                    if s != name:
                        name = f"{name} ({s})"
                        break

            if name in disambiguated_aliases:
                if disambiguated_aliases[name] != cui:
                        failed += 1
                        alt_ids2cui[cui] = disambiguated_aliases[name]
                        #print(f"Failed to disambiguate:\nEntity 1. {cui}\nName: {name}\nMeta:{entity}\nEntity 2. {disambiguated_aliases[name]}\nMeta: {kb[disambiguated_aliases[name]]}\n\n")
            else:
                disambiguated_aliases[name] = cui

    print(f"Total homonyms resolved: {homonyms}")
    print(f"Failed to resolve {failed} homonyms :(")
    print(f"Final unique names: {len(disambiguated_aliases)}")
    print(f"Avg. names per cui: {sum(avg_names)/len(avg_names):.2f}" )

    statistics = {
            "num_cuis": len(kb),
            "num_names": len(disambiguated_aliases),
            "num_homonyms":f"{homonyms} ({homonyms/len(disambiguated_aliases):.2f})",
            "num_failed": f"{failed} ({failed/len(disambiguated_aliases):.2f})",
            "avg_names_x_cui": f"{sum(avg_names)/len(avg_names):.2f}"
        }
    return disambiguated_aliases, alt_ids2cui, statistics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="belink-hpo prepare-obo", description="Turn an OBO ontology into the list of names for the retrieval index")
    parser.add_argument("--ontology_path", type=str, required=True,
                        help="Path to .obo file (e.g., ../datasets/ncbitaxon.obo) or a URL")
    parser.add_argument("--output_dir", type=str,
                        default="obo_ontology", 
                        help="Path of output directory")
    parser.add_argument("--cache_obo", action="store_true", 
                        help="Whether to save .obo to disk after download. Default: False.")
    
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    filepath = args.ontology_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # down(load) and parse the .obo file 
    onto = load_obo(filepath, cache_obo=args.cache_obo)

    # disambigate ontology aliases -> final terminology
    name2cui, new_kb, alt_ids2cui = get_name2cuis_mapping(onto)
    disambiguated_names, alt_ids2cui, statistics = resolve_homonyms(name2cui, new_kb, alt_ids2cui)

    excluded_cuis = ["hp:0000001"]  # the root term "All"; ids are lower-case at this point
    disambiguated_kb = []
    for name, cui in disambiguated_names.items():
        if name.strip() and cui.strip():
            clean_name = re.sub(r"\n", " ", name.strip())
        if cui not in excluded_cuis:
            disambiguated_kb.append({"id": cui.strip(), "name": clean_name})

    with gzip.open(output_dir /"processed_obo.json.gz", "wt", encoding="utf-8") as file:
        json.dump(onto, file, indent=4)

    with gzip.open(output_dir /"terminology.json.gz", "wt", encoding="utf-8") as file:
        json.dump(disambiguated_kb, file, indent=4)

    with open(output_dir / "alt_ids2cui.json", "w") as file:
        json.dump(alt_ids2cui, file, indent=4)
    
    with open(output_dir / "statistics_kb.json", "w") as file:
        json.dump(statistics, file, indent=4)

if __name__ == "__main__":
    main()
