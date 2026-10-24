"""
Knowledge graph from Wikidata: the graph shown in the web interface and the arbiter's KG evidence.

Every node is a Wikidata entity (or a literal value such as a date) and every edge a Wikidata
statement, so the graph needs no LLM call. The claim itself is drawn as an edge from its subject
to the claimed value, next to what Wikidata records for the matching property.

The detector builds it in a background thread as soon as query generation has named the claim's
entities. The web interface draws it, and for an escalated claim its statements (as_evidence) are
added to the LLM arbiter's evidence.
"""
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://www.wikidata.org/w/api.php"
# Wikidata's API policy asks for a descriptive User-Agent
USER_AGENT = "VeriFact/1.0 (academic research prototype; https://github.com/Raj-IIT-Kgp/Hallucination-Detector)"
TIMEOUT_SEC = 8

# Curated properties that claims commonly state: pid -> (label, phrase used for matching)
PROPERTIES = {
    "P569": ("date of birth", "born in, birth date, birthday, year of birth"),
    "P570": ("date of death", "died in, death date, year of death"),
    "P19": ("place of birth", "born in a city or country, birthplace"),
    "P20": ("place of death", "died in a place"),
    "P27": ("country of citizenship", "nationality, citizen of a country, is American, is British, is Indian"),
    "P106": ("occupation", "profession, works as, is an actor, is a writer, is a producer, is a singer, is a politician"),
    "P101": ("field of work", "works in the field of"),
    "P166": ("award received", "won, won an award, won a prize, received an honour, awarded"),
    "P1411": ("nominated for", "was nominated for an award, nomination"),
    "P26": ("spouse", "married to, wife, husband"),
    "P22": ("father", "father of, son or daughter of"),
    "P25": ("mother", "mother of"),
    "P40": ("child", "children, son, daughter"),
    "P3373": ("sibling", "brother, sister"),
    "P69": ("educated at", "studied at, attended university or school, graduated from"),
    "P184": ("doctoral advisor", "PhD supervisor"),
    "P108": ("employer", "worked for, employed by"),
    "P39": ("position held", "served as, office, title"),
    "P102": ("member of political party", "party membership"),
    "P463": ("member of", "member of an organization"),
    "P54": ("member of sports team", "played for team, club"),
    "P413": ("position played on team", "plays as position"),
    "P800": ("notable work", "known for work, famous for"),
    "P50": ("author", "written by, wrote"),
    "P57": ("director", "directed by"),
    "P58": ("screenwriter", "screenplay written by"),
    "P161": ("cast member", "starring, starred in the film, appeared in the cast"),
    "P86": ("composer", "music composed by"),
    "P175": ("performer", "performed by, singer, band"),
    "P162": ("producer", "produced by"),
    "P272": ("production company", "produced by company, studio"),
    "P449": ("original broadcaster", "aired on network, channel"),
    "P136": ("genre", "genre, kind of film, music or book"),
    "P495": ("country of origin", "made in country, originated in, is an American film, is a British series, is Swedish"),
    "P364": ("original language of film or TV show", "language of the film or show"),
    "P577": ("publication date", "released, published, premiered in"),
    "P112": ("founded by", "founder, co-founder, established by"),
    "P571": ("inception", "founded in, established in, created in year"),
    "P576": ("dissolved, abolished or demolished date", "ended, closed, dissolved in"),
    "P159": ("headquarters location", "headquartered in, based in"),
    "P169": ("chief executive officer", "CEO of"),
    "P488": ("chairperson", "chairman, chair"),
    "P127": ("owned by", "owner, owned by, acquired by"),
    "P749": ("parent organization", "subsidiary of, parent company"),
    "P452": ("industry", "business sector"),
    "P17": ("country", "located in country"),
    "P131": ("located in the administrative territorial entity", "located in city, state, region"),
    "P276": ("location", "located at, takes place in"),
    "P30": ("continent", "continent"),
    "P36": ("capital", "capital city of"),
    "P1376": ("capital of", "is the capital of"),
    "P35": ("head of state", "president, monarch, king, queen"),
    "P6": ("head of government", "prime minister, mayor, governor"),
    "P37": ("official language", "language spoken officially"),
    "P38": ("currency", "currency, money"),
    "P1082": ("population", "number of inhabitants, people living"),
    "P2046": ("area", "area, size in square kilometres"),
    "P2044": ("elevation above sea level", "height above sea level, altitude"),
    "P2048": ("height", "how tall, height"),
    "P2043": ("length", "how long, length"),
    "P84": ("architect", "designed by architect"),
    "P61": ("discoverer or inventor", "discovered by, invented by"),
    "P138": ("named after", "named after, name comes from"),
    "P176": ("manufacturer", "made by, manufactured by, built by"),
    "P619": ("UTC date of spacecraft launch", "launched in, launch date"),
    "P585": ("point in time", "happened in, date of the event"),
    "P580": ("start time", "began in, started in"),
    "P582": ("end time", "ended in"),
    "P1344": ("participant in", "took part in, competed in"),
    "P140": ("religion or worldview", "religion, faith"),
    "P509": ("cause of death", "died of"),
    "P1412": ("languages spoken, written or signed", "speaks language"),
    "P264": ("record label", "signed to label"),
    "P1303": ("instrument", "plays instrument"),
    "P361": ("part of", "part of, belongs to"),
    "P527": ("has part(s)", "consists of, includes"),
    "P31": ("instance of", "is a kind of thing, type"),
}
# Properties whose values are people: a claimed value for them must be a person, and for the others must not
HUMAN_VALUED = {"P22", "P25", "P40", "P26", "P3373", "P184", "P50", "P57", "P58", "P161", "P86", "P162",
                "P112", "P169", "P488", "P35", "P6", "P84", "P61", "P175"}
HUMAN = "Q5"
QUANTITY_PROPERTIES = {"P1082", "P2046", "P2044", "P2048", "P2043"}
# "Located in" can be recorded at several levels; a claimed place agrees if any of them lists it
LOCATION_FAMILY = ["P276", "P131", "P17", "P159"]
TIME_PROPERTIES = {"P569", "P570", "P571", "P576", "P577", "P619", "P585", "P580", "P582"}
# Properties with one true value, so a different claimed value contradicts the record. Places are
# left out: they nest (a person born in Ulm was also born in Germany), so an unlisted place is only
# "not listed", not a contradiction.
SINGLE_VALUED = {"P569", "P570", "P571", "P576", "P577", "P619", "P585", "P22", "P25", "P36"}

# A claim's relation must reach this score (cosine similarity to the property's best phrase, plus
# the adjustments in build_graph) to count as a match
MIN_PROPERTY_SIM = 0.45
CONTEXT_EDGES = 4           # context statements shown per claim
OTHER_CONTEXT_EDGES = 1     # of which at most this many about the claim's other entities
VALUES_PER_PROPERTY = 3
TYPE_VALUES = 2             # "instance of" values of the subject, always included
CANDIDATE_PROPERTIES = 10   # per entity, before value labels are fetched
DISAMBIGUATION_PAGE = "Q4167410"

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]
_MONTH_RE = "|".join(m[:3] + r"[a-z]*\.?" for m in _MONTHS)
_DATE_RE = re.compile(
    rf"\b(?:(\d{{1,2}})\s+({_MONTH_RE})\s+(\d{{3,4}})"        # 14 March 1879
    rf"|({_MONTH_RE})\s+(\d{{1,2}}),?\s+(\d{{3,4}})"           # March 14, 1879
    rf"|({_MONTH_RE})\s+(\d{{3,4}})"                           # March 1879
    rf"|(\d{{4}})s?)\b", re.IGNORECASE)                         # 1879
_NUMBER_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)(?:\s+(thousand|million|billion|trillion))?\b", re.IGNORECASE)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_STOP = {"the", "a", "an", "of", "in", "on", "at", "and", "or", "to", "for", "by", "with", "from",
         "is", "was", "are", "were", "be", "been", "his", "her", "its", "their", "list"}

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_cache_lock = threading.Lock()
_entity_cache = {}      # name.lower() -> entity dict, or None when it could not be resolved
_label_cache = {}       # QID -> English label
_CACHE_MAX = 2000
_property_embs = None
# Sentence encoder (all-MiniLM-L6-v2), loaded on first use; the lock keeps threads from calling the
# model at the same time (macOS malloc double-free protection)
_encoder = None
_encoder_lock = threading.Lock()
# Separate pools: a graph build waits on its entity lookups, so they must not share workers
_build_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wikidata-graph")
_fetch_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="wikidata-fetch")


def _content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def _api(params):
    params = {**params, "format": "json", "formatversion": 2}
    response = _session.get(API, params=params, timeout=TIMEOUT_SEC)
    if response.status_code == 429:
        # Rate limited: one retry after the requested pause (capped, since an escalated claim may wait)
        try:
            pause = float(response.headers.get("Retry-After", 2))
        except ValueError:
            pause = 2
        time.sleep(min(max(pause, 1), 5))
        response = _session.get(API, params=params, timeout=TIMEOUT_SEC)
    response.raise_for_status()
    return response.json()


def _remember(cache, key, value):
    with _cache_lock:
        if len(cache) >= _CACHE_MAX:
            cache.clear()
        cache[key] = value


def _entity_from_api(data):
    """Keeps what the graph needs from a wbgetentities entity."""
    if not data or "missing" in data:
        return None
    claims = data.get("claims", {})
    instance_of = {s["mainsnak"].get("datavalue", {}).get("value", {}).get("id") for s in claims.get("P31", [])}
    if DISAMBIGUATION_PAGE in instance_of:
        return None
    statements = {}
    for pid in PROPERTIES:
        values = []
        ranked = sorted(claims.get(pid, []), key=lambda s: s.get("rank") != "preferred")
        for s in ranked:
            snak = s.get("mainsnak", {})
            if s.get("rank") == "deprecated" or snak.get("snaktype") != "value":
                continue
            value = _parse_value(snak.get("datavalue", {}))
            if value:
                value["refs"] = len(s.get("references", []))
                values.append(value)
        if values:
            statements[pid] = values
    title = data.get("sitelinks", {}).get("enwiki", {}).get("title")
    label = data.get("labels", {}).get("en", {}).get("value") or title or data["id"]
    return {
        "id": data["id"],
        "label": label,
        "description": data.get("descriptions", {}).get("en", {}).get("value", ""),
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title
               else f"https://www.wikidata.org/wiki/{data['id']}",
        "statements": statements,
        "human": HUMAN in instance_of,
    }


def _parse_value(datavalue):
    kind, v = datavalue.get("type"), datavalue.get("value")
    if kind == "wikibase-entityid":
        return {"type": "item", "id": v.get("id")}
    if kind == "time":
        return {"type": "time", "time": v.get("time", ""), "precision": v.get("precision", 9)}
    if kind == "quantity":
        unit = v.get("unit", "1")
        return {"type": "quantity", "amount": v.get("amount", ""), "unit": unit.rsplit("/", 1)[-1] if unit != "1" else None}
    if kind == "string":
        return {"type": "string", "text": v}
    if kind == "monolingualtext":
        return {"type": "string", "text": v.get("text", "")}
    return None


def resolve_entity(name):
    """
    Resolves an entity name to a Wikidata item: the English Wikipedia page of that exact title
    (redirects followed); otherwise one search, accepted only if the result's label shares a
    content word with the name and, for a qualified title such as "Colombiana (2011 film)", its
    description shares a word with the qualifier. Returns None instead of guessing a near match.
    """
    key = name.lower().strip()
    with _cache_lock:
        if key in _entity_cache:
            return _entity_cache[key]
    props = "labels|descriptions|claims|sitelinks"
    entity = None
    data = _api({"action": "wbgetentities", "sites": "enwiki", "titles": name, "normalize": 1,
                 "redirects": "yes", "props": props, "languages": "en", "sitefilter": "enwiki"})
    found = [e for e in (data.get("entities") or {}).values() if "missing" not in e]
    if found:
        entity = _entity_from_api(found[0])
    else:
        m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", name)
        base, qualifier = (m.group(1), m.group(2)) if m else (name, "")
        hits = _api({"action": "wbsearchentities", "search": base, "language": "en", "type": "item",
                     "limit": 5}).get("search", [])
        words, qualifier_words = _content_words(base), _content_words(qualifier) - {"disambiguation"}
        for hit in hits:
            description_words = _content_words(hit.get("description", ""))
            if words & _content_words(hit.get("label", "")) and (not qualifier_words or qualifier_words & description_words):
                data = _api({"action": "wbgetentities", "ids": hit["id"], "props": props,
                             "languages": "en", "sitefilter": "enwiki"})
                entity = _entity_from_api((data.get("entities") or {}).get(hit["id"]))
                break
    _remember(_entity_cache, key, entity)
    return entity


def _fetch_labels(qids):
    """English labels for item values and units, 50 per request."""
    missing = [q for q in dict.fromkeys(qids) if q and q not in _label_cache]
    for i in range(0, len(missing), 50):
        batch = missing[i:i + 50]
        data = _api({"action": "wbgetentities", "ids": "|".join(batch), "props": "labels", "languages": "en"})
        for q in batch:
            e = (data.get("entities") or {}).get(q, {})
            _remember(_label_cache, q, e.get("labels", {}).get("en", {}).get("value") or q)
    return {q: _label_cache.get(q, q) for q in qids if q}


def _format_time(value):
    m = re.match(r"([+-])(\d+)-(\d\d)-(\d\d)", value["time"])
    if not m:
        return value["time"]
    sign, year, month, day = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    era = " BCE" if sign == "-" else ""
    precision = value["precision"]
    if precision >= 11 and month and day:
        return f"{day} {_MONTHS[month - 1]} {year}{era}"
    if precision == 10 and month:
        return f"{_MONTHS[month - 1]} {year}{era}"
    if precision == 8:
        return f"{year // 10 * 10}s{era}"
    if precision == 7:
        return f"{(year - 1) // 100 + 1}th century{era}"
    return f"{year}{era}"


def _format_value(value, labels):
    if value["type"] == "item":
        return labels.get(value["id"], value["id"])
    if value["type"] == "time":
        return _format_time(value)
    if value["type"] == "quantity":
        amount = value["amount"].lstrip("+")
        try:
            number = float(amount)
            amount = f"{number:,.0f}" if number.is_integer() else f"{number:,}"
        except ValueError:
            pass
        unit = labels.get(value["unit"], "") if value["unit"] else ""
        return f"{amount} {unit}".strip()
    return value.get("text", "")


def _value_node_id(v, labels):
    return v["id"] if v["type"] == "item" else f"value:{_format_value(v, labels)}"


def _get_encoder():
    global _encoder
    if _encoder is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _encoder


def _encode(texts):
    with _encoder_lock:
        return _get_encoder().encode(texts, show_progress_bar=False, normalize_embeddings=True)


def _property_embeddings():
    """Embeddings of each property's label and of each of its phrases, with the owning property of each row."""
    global _property_embs
    if _property_embs is None:
        owners, texts = [], []
        for pid, (label, phrases) in PROPERTIES.items():
            for text in [label] + [t.strip() for t in phrases.split(",")]:
                owners.append(pid)
                texts.append(text)
        _property_embs = (owners, _encode(texts))
    return _property_embs


def _property_similarities(relation_emb):
    """Each property's similarity to the relation: the best of its label and phrases."""
    owners, embs = _property_embeddings()
    best = {}
    for pid, sim in zip(owners, embs @ relation_emb):
        best[pid] = max(best.get(pid, -1.0), float(sim))
    return best


def _relation_text(claim, names):
    """The claim with its entity names removed, so similarity reflects the relation."""
    text = claim
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" .,") or claim


def _mentioned(entity, names, claim):
    """Whether the claim itself names the entity (queries can also name entities from the context)."""
    lower = claim.lower()
    for text in [entity["label"]] + names:
        text = re.sub(r"\s*\([^)]*\)", "", text.lower())
        text = re.sub(r",?\s+(inc|ltd|llc|plc|corp)\.?$", "", text).strip()
        if text and text in lower:
            return True
    return False


def _claimed_dates(claim):
    """(year, month, day) tuples stated in the claim; month and day are None when not stated."""
    out = []
    for m in _DATE_RE.finditer(claim):
        if m.group(1):
            out.append((int(m.group(3)), _month(m.group(2)), int(m.group(1))))
        elif m.group(4):
            out.append((int(m.group(6)), _month(m.group(4)), int(m.group(5))))
        elif m.group(7):
            out.append((int(m.group(8)), _month(m.group(7)), None))
        elif m.group(9):
            out.append((int(m.group(9)), None, None))
    return out


def _month(text):
    t = text[:3].lower()
    return next((i + 1 for i, m in enumerate(_MONTHS) if m[:3].lower() == t), None)


def _time_parts(value):
    m = re.match(r"[+-](\d+)-(\d\d)-(\d\d)", value["time"])
    if not m:
        return None
    p = value["precision"]
    return int(m.group(1)), (int(m.group(2)) if p >= 10 else None), (int(m.group(3)) if p >= 11 else None)


def _date_agrees(claimed, recorded):
    if claimed[0] != recorded[0]:
        return False
    return all(c is None or r is None or c == r for c, r in zip(claimed[1:], recorded[1:]))


def _compare(pid, claim, recorded, labels, other_entities):
    """
    Compares the claim with the recorded values of the matched property.
    Returns (claimed value text or None, status, matching recorded value text or None).
    Status: agrees / differs / not_listed (not among the recorded values) / unclear.
    """
    lower = claim.lower()
    texts = [_format_value(v, labels) for v in recorded]
    # Only an entity of the right kind can be the claimed value (an event is never an architect)
    other_entities = [o for o in other_entities if o.get("human", False) == (pid in HUMAN_VALUED)]
    if pid in TIME_PROPERTIES:
        dates = _claimed_dates(claim)
        if not dates:
            return None, "unclear", None
        claimed = dates[0]
        claimed_text = " ".join(str(x) for x in [claimed[2], _MONTHS[claimed[1] - 1] if claimed[1] else None, claimed[0]] if x)
        for v, t in zip(recorded, texts):
            parts = _time_parts(v) if v["type"] == "time" else None
            if parts and _date_agrees(claimed, parts):
                return claimed_text, "agrees", t
        return claimed_text, "differs", None
    if recorded[0]["type"] == "item":
        ids = {v["id"] for v in recorded}
        for other in other_entities:
            if other["id"] in ids:
                return other["label"], "agrees", other["label"]
        for t in texts:
            if t and t.lower() in lower:
                return t, "agrees", t
        if other_entities:
            return other_entities[0]["label"], "differs" if pid in SINGLE_VALUED else "not_listed", None
        return None, "unclear", None
    if recorded[0]["type"] == "quantity":
        # (value, tolerance, text): "14 million" is a rounded figure, so it gets a looser tolerance
        numbers = [(float(n.replace(",", "")) * _SCALE.get(scale.lower(), 1), 0.05 if scale else 0.01,
                    f"{n} {scale}".strip()) for n, scale in _NUMBER_RE.findall(claim)]
        if not numbers:
            return None, "unclear", None
        for v, t in zip(recorded, texts):
            try:
                amount = float(v["amount"])
            except ValueError:
                continue
            if any(abs(n - amount) <= tol * max(abs(amount), 1) for n, tol, _ in numbers):
                return t, "agrees", t
        return numbers[0][2], "differs", None
    for t in texts:
        if t and t.lower() in lower:
            return t, "agrees", t
    return None, "unclear", None


def build_graph(claim, entity_names):
    """
    Builds the Wikidata graph for one claim from the entity names named by query generation.
    Returns {"source": "wikidata", "nodes", "edges", "match", "unresolved"}; raises on network errors.
    """
    names = [n for n in dict.fromkeys(n.strip() for n in entity_names or []) if n and n.lower() != claim.lower()][:3]
    if not names:
        return {"source": "wikidata", "nodes": [], "edges": [], "match": None, "unresolved": []}
    resolved = list(_fetch_pool.map(resolve_entity, names)) if len(names) > 1 else [resolve_entity(names[0])]
    entities, unresolved, seen = [], [], set()
    for name, e in zip(names, resolved):
        if e is None:
            unresolved.append(name)
        elif e["id"] not in seen:
            seen.add(e["id"])
            entities.append(e)
    if not entities:
        return {"source": "wikidata", "nodes": [], "edges": [], "match": None, "unresolved": unresolved}

    # Subject: the resolved entity whose label or name appears earliest in the claim
    lower = claim.lower()
    def position(e):
        found = [lower.find(s.lower()) for s in [e["label"]] + names if s.lower() in lower]
        return min(found) if found else len(claim)
    entities.sort(key=position)
    subject, others = entities[0], entities[1:]
    # Only entities the claim itself names can be its claimed value
    names_of = {e["id"]: [n for n, r in zip(names, resolved) if r is not None and r["id"] == e["id"]] for e in entities}
    others = [o for o in others if _mentioned(o, names_of[o["id"]], claim)]

    # Match the claim's relation to a property: similarity of the claim without its subject to the
    # property phrases, adjusted for the kind of value the claim states and what the subject records
    subject_names = [subject["label"]] + [n for n, e in zip(names, resolved) if e is not None and e["id"] == subject["id"]]
    # Two readings of the relation: without the subject (keeps a telling object such as "Nobel
    # Prize"), and without every entity name (so an object such as "European Space Agency" does
    # not swamp "was built by"); each property takes the better of the two
    all_names = names + [e["label"] for e in entities]
    relation_emb, bare_emb = _encode([_relation_text(claim, subject_names), _relation_text(claim, all_names)])
    sims = _property_similarities(relation_emb)
    if others:
        for pid, sim in _property_similarities(bare_emb).items():
            sims[pid] = max(sims[pid], sim)
    states_date = bool(_claimed_dates(claim)) and not others
    scored = []
    for pid, score in sims.items():
        if pid in TIME_PROPERTIES:
            score += 0.12 if states_date else -0.08
        if any(pid in e["statements"] for e in entities):
            score += 0.10
        scored.append((score, pid))
    scored.sort(reverse=True)
    # A claim that names a second entity states an entity as its value, not a date or number
    eligible = [(sc, pid) for sc, pid in scored
                if not (others and (pid in TIME_PROPERTIES or pid in QUANTITY_PROPERTIES))]
    best_score, best_pid = eligible[0]
    matched_pid = best_pid if best_score >= MIN_PROPERTY_SIM and best_pid != "P31" else None
    prop_rank = {pid: k for k, (_, pid) in enumerate(scored)}

    # The entity that records the claim's property: usually the subject, but some statements are
    # stored the other way round (Apple records its CEO, not Tim Cook his employer's title)
    holder, holder_others = subject, others
    if matched_pid and matched_pid not in subject["statements"]:
        for o in others:
            if matched_pid in o["statements"]:
                holder, holder_others = o, [subject] + [x for x in others if x is not o]
                break

    # Candidate statements: the claim's property plus the properties closest to the claim
    candidates = []   # (entity, pid, value)
    for e in entities:
        props = sorted(e["statements"], key=lambda p: prop_rank[p])[:CANDIDATE_PROPERTIES]
        if matched_pid and matched_pid in e["statements"] and matched_pid not in props:
            props.append(matched_pid)
        for pid in props:
            for v in e["statements"][pid][:VALUES_PER_PROPERTY]:
                candidates.append((e, pid, v))
    claim_pids = ([matched_pid] + (LOCATION_FAMILY if matched_pid in LOCATION_FAMILY else [])) if matched_pid else []
    claim_values = [v for pid in claim_pids for v in holder["statements"].get(pid, [])[:25]]
    # What the subject is ("instance of": street gang, film, human), the deciding fact for "X is a Y" claims
    type_values = holder["statements"].get("P31", [])[:TYPE_VALUES]
    labels = _fetch_labels([v.get("id") for _, _, v in candidates if v["type"] == "item"] +
                           [v.get("id") for v in claim_values + type_values if v["type"] == "item"] +
                           [v.get("unit") for _, _, v in candidates if v["type"] == "quantity"])

    nodes, edges = {}, []
    def entity_node(e):
        nodes.setdefault(e["id"], {"id": e["id"], "label": e["label"], "kind": "entity", "url": e["url"],
                                   "summary_snippet": e["description"], "qid": e["id"]})
        return e["id"]
    def value_node(pid, v):
        if v["type"] == "item":
            node_id = v["id"]
            nodes.setdefault(node_id, {"id": node_id, "label": labels.get(node_id, node_id), "kind": "item",
                                       "url": f"https://www.wikidata.org/wiki/{node_id}", "qid": node_id})
        else:
            text = _format_value(v, labels)
            node_id = f"value:{text}"
            nodes.setdefault(node_id, {"id": node_id, "label": text, "kind": "value"})
        return node_id
    def statement_edge(e, pid, v, role, agrees=False):
        edges.append({"source": entity_node(e), "target": value_node(pid, v), "relationship": PROPERTIES[pid][0],
                      "pid": pid, "kind": role, "refs": v["refs"], "agrees": agrees,
                      "url": f"https://www.wikidata.org/wiki/{e['id']}#{pid}"})
    for e in entities:
        entity_node(e)

    # The claim edge and the record for its property
    match = {"property": None, "pid": None, "score": round(best_score, 3), "status": "unmatched",
             "claimed": None, "recorded": [], "agreeing": None}
    if matched_pid:
        recorded = holder["statements"].get(matched_pid, [])
        if recorded:
            claimed, status, agreeing = _compare(matched_pid, claim, recorded, labels, holder_others)
            if status != "agrees" and matched_pid in LOCATION_FAMILY:
                for pid in LOCATION_FAMILY:
                    values = holder["statements"].get(pid, [])
                    if pid != matched_pid and values:
                        result = _compare(pid, claim, values, labels, holder_others)
                        if result[1] == "agrees":
                            matched_pid, recorded, (claimed, status, agreeing) = pid, values, result
                            break
        match.update(pid=matched_pid, property=PROPERTIES[matched_pid][0])
        if recorded:
            if len(recorded) > VALUES_PER_PROPERTY:
                # Of many values (e.g. awards), show those closest to the claim
                value_sims = _encode([_format_value(v, labels) for v in recorded[:25]]) @ relation_emb
                recorded = [recorded[k] for k in sorted(range(len(value_sims)), key=lambda k: -value_sims[k])] + recorded[25:]
            shown_values = recorded[:VALUES_PER_PROPERTY]
            agreeing_value = next((v for v in recorded if _format_value(v, labels) == agreeing), None)
            if agreeing_value is not None and agreeing_value not in shown_values:
                shown_values = shown_values[:-1] + [agreeing_value]
            match.update(status=status, claimed=claimed, agreeing=agreeing,
                         recorded=[_format_value(v, labels) for v in shown_values], total_recorded=len(recorded))
            for v in shown_values:
                statement_edge(holder, matched_pid, v, "record", agrees=v is agreeing_value)
        else:
            match["status"] = "not_in_wikidata"
            fitting = [o for o in holder_others if o.get("human", False) == (matched_pid in HUMAN_VALUED)]
            claimed = fitting[0]["label"] if fitting else None
            match["claimed"] = claimed
        if claimed and match["status"] != "agrees":
            target = next((o["id"] for o in holder_others if o["label"] == claimed), None) or f"claimed:{claimed}"
            if target not in nodes:
                nodes[target] = {"id": target, "label": claimed, "kind": "claimed"}
            edges.append({"source": holder["id"], "target": target, "relationship": PROPERTIES[matched_pid][0],
                          "pid": matched_pid, "kind": "claim", "status": match["status"]})

    # Direct statements between the claim's entities, then the context closest to the claim
    shown = {(edge["source"], edge["target"], edge["pid"]) for edge in edges}
    ids = {e["id"] for e in entities}
    for e, pid, v in candidates:
        if v["type"] == "item" and v["id"] in ids and v["id"] != e["id"] and (e["id"], v["id"], pid) not in shown:
            statement_edge(e, pid, v, "link")
            shown.add((e["id"], v["id"], pid))
    context, keys = [], set()
    for e, pid, v in candidates:
        key = (e["id"], _value_node_id(v, labels), pid)
        if pid != "P31" and key not in shown and key not in keys:
            keys.add(key)
            context.append((e, pid, v))
    # Context only about entities joined to the subject, so the graph stays one connected picture
    connected, frontier = {holder["id"]}, [holder["id"]]
    while frontier:
        current = frontier.pop()
        for edge in edges:
            for a, b in ((edge["source"], edge["target"]), (edge["target"], edge["source"])):
                if a == current and b not in connected:
                    connected.add(b)
                    frontier.append(b)
    context = [(e, pid, v) for e, pid, v in context if e["id"] in connected]
    if context:
        # Ranked against the relation, not the whole claim, so the subject's name does not dominate
        texts = [f"{PROPERTIES[pid][0]}: {_format_value(v, labels)}" for e, pid, v in context]
        sims = _encode(texts) @ relation_emb
        order = sorted(range(len(context)), key=lambda k: -sims[k])
        added, others_added, per_prop = 0, 0, {}
        for k in order:
            e, pid, v = context[k]
            if added >= CONTEXT_EDGES:
                break
            if per_prop.get((e["id"], pid), 0) >= 2:
                continue
            if e is not holder:
                if others_added >= OTHER_CONTEXT_EDGES:
                    continue
                others_added += 1
            per_prop[(e["id"], pid)] = per_prop.get((e["id"], pid), 0) + 1
            statement_edge(e, pid, v, "context")
            added += 1

    for v in type_values:
        if (holder["id"], _value_node_id(v, labels), "P31") not in shown:
            statement_edge(holder, "P31", v, "context")

    used = {holder["id"]} | {edge["source"] for edge in edges} | {edge["target"] for edge in edges}
    return {"source": "wikidata", "claim": claim, "subject": holder["id"],
            "nodes": [n for n in nodes.values() if n["id"] in used],
            "edges": edges, "match": match, "unresolved": unresolved}


WIKIDATA_NOTE = ("Wikidata Evidence: structured statements recorded in Wikidata, with their number of references. "
                 "Wikidata can be incomplete or out of date.")


def as_evidence(graph):
    """
    The graph's Wikidata statements as evidence text for the LLM arbiter, or "" if there are none.
    The claim edge is left out: it is what the claim states, not evidence.
    """
    labels = {n["id"]: n["label"] for n in (graph or {}).get("nodes", [])}
    lines = []
    for e in (graph or {}).get("edges", []):
        if e["kind"] == "claim":
            continue
        refs = e.get("refs", 0)
        source, target = labels[e["source"]], labels[e["target"]]
        lines.append(f"Wikidata Evidence [{source}]: {source} — {e['relationship']}: {target} "
                     f"({refs} reference{'' if refs == 1 else 's'} in Wikidata).")
    return "\n\n".join([WIKIDATA_NOTE] + lines) if lines else ""


def start(claim, entity_names):
    """Starts building the graph in the background; returns a Future."""
    return _build_pool.submit(build_graph, claim, entity_names)
