import argparse
import json
import re
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import jiwer
import pandas as pd
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import normalize as K
RK = K
from hf_normalizer.normalizer import EnglishTextNormalizer

EDGE = ".,;:!?\"'()[]{}"
GLUE = "-/—–,."
FILLERS = {"at", "the", "on", "of", "and", "in", "a", "to", "is", "for", "by", "or"}
WORD = re.compile(r"\S+")


def clean(text):
    return re.sub(r"\s+", " ", K.strip_artifacts(text if isinstance(text, str) else "")).strip()


def words(text):
    return [(m.start(), m.end(), re.sub(r"[^\w]", "", m.group()).lower()) for m in WORD.finditer(text)]


def entity_ranges(ref, ents):
    w, out, pos = words(ref), [], 0
    for e in ents:
        i = ref.find(e, pos)
        i = ref.find(e) if i < 0 else i
        if i < 0:
            out.append(None)
            continue
        j = i + len(e)
        idx = [k for k, (s, t, _) in enumerate(w) if s < j and t > i]
        out.append((idx[0], idx[-1]))
        pos = j
    return out


def spans_for(ref, hyp, ranges):
    rw, hw = words(ref), words(hyp)
    eq = {}
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes([x[2] for x in rw], [x[2] for x in hw]):
        if tag == "equal":
            for k in range(i2 - i1):
                eq[i1 + k] = j1 + k
    out = []
    for rg in ranges:
        if not rg:
            out.append(None)
            continue
        a, b = rg
        left = next((eq[k] for k in range(a - 1, -1, -1) if k in eq), -1)
        right = next((eq[k] for k in range(b + 1, len(rw)) if k in eq), len(hw))
        out.append((hw[left + 1][0], hw[right - 1][1]) if right > left + 1 else None)
    return out


def _boundaries(text):
    pos = {0, len(text)}
    for m in re.finditer(r"\s+", text):
        pos |= {m.start(), m.end()}
    for i, ch in enumerate(text):
        if ch in GLUE and 0 < i < len(text) - 1:
            pos |= {i, i + 1}
    return sorted(pos)


NUM_TOKEN = re.compile(r"^[\s,\-.]*(\d|(?:%s)\b)" % "|".join(sorted(K.NUMBER_WORDS | {"oh", "hundred"}, key=len, reverse=True)))
NUM_TAIL = re.compile(r"(\d|\b(?:%s))[\s,\-]*$" % "|".join(sorted(K.NUMBER_WORDS | {"oh", "hundred"}, key=len, reverse=True)))


def _number_continues(text, i, j, entity):
    if re.search(r"\d$", entity.strip()) and NUM_TAIL.search(text[i:j].rstrip(EDGE + " ")) and NUM_TOKEN.match(text[j:]):
        return True
    if re.match(r"\d", entity.strip()) and NUM_TAIL.search(text[:i]) and re.match(r"[\s]*(\d|(?:%s)\b)" % "|".join(
            K.NUMBER_WORDS), text[i:]) and not re.search(r"[a-z]\s*$", text[:i].rstrip(" ,-")[-1:] or " "):
        return True
    return False


def _neighbour_ok(text, i, j, form, entity):
    seg = text[i:j]
    j = i + len(seg.rstrip(EDGE + GLUE + " "))
    i = i + (len(seg) - len(seg.lstrip(EDGE + GLUE + " ")))
    after, before = text[j:j + 1], text[i - 1:i] if i else ""
    if _number_continues(text, i, j, entity):
        return False
    if form == "number_amount" and after == "/" and "$" not in entity:
        return False
    if form in ("alnum_code", "digit_sequence"):
        for ch, nb in ((after, re.match(r"[\w]+", text[j + 1:])), (before, re.search(r"[\w]+$", text[:max(0, i - 1)]))):
            if ch and ch in GLUE and nb and not (nb.group().isalpha() and len(nb.group()) >= 3):
                return False
    return True


def _is_filler(piece):
    w = piece.strip().strip(EDGE + GLUE)
    return w in FILLERS or (piece.strip() and piece.strip()[0] in GLUE and w.isalpha() and len(w) > 1)


def best_rendering(span, entity, form, possessive=False, same=K.same_entity, longest=0):
    text = span
    if not text:
        return "", -1
    n = len(entity.split())
    bd = _boundaries(text)
    max_chars = max(4 * len(entity) + 30, longest + 10)

    def cl(i, j):
        c = text[i:j].strip().strip(EDGE + GLUE.replace(".", "").replace(",", "")).strip()
        return re.sub(r"'s$", "", c) if possessive else c

    def ok(i, j):
        c = cl(i, j)
        return bool(c) and same(c, entity, form) and _neighbour_ok(text, i, j, form, entity)

    pairs = []
    for x, i in enumerate(bd):
        for j in bd[x + 1:]:
            if j - i > max_chars:
                break
            pairs.append((i, j))
    for i, j in sorted(pairs, key=lambda p: (p[1] - p[0], p[0])):
        if not ok(i, j):
            continue
        a, b = bd.index(i), bd.index(j)
        while True:
            if a > 0 and ok(bd[a - 1], bd[b]) and not _is_filler(text[bd[a - 1]:bd[a]]):
                a -= 1
            elif b < len(bd) - 1 and ok(bd[a], bd[b + 1]) and not _is_filler(text[bd[b]:bd[b + 1]]):
                b += 1
            else:
                break
        c = cl(bd[a], bd[b])
        return c, text.find(c, bd[a])
    return "", -1


def _match_job(args):
    span, entity, form, possessive = args
    return best_rendering(span, entity, form, possessive)


def variant_patterns(emap):
    return {k: re.compile(r"(?<![\w@.])(?:%s)(?![\w@])" % "|".join(
        re.escape(v) for v in sorted(x["variants"], key=len, reverse=True))) for k, x in emap.items() if x["variants"]}


def find_entity(e, s, hyp, same, pats, reject, longest=0):
    span = hyp[s[0]:s[1]] if s else ""
    rend, off = best_rendering(span, e["entity"], e["form"], e["possessive"], same, longest) if span else ("", -1)
    if rend and rend not in reject.get(e["entity_id"], ()):
        return rend, "aligned", s[0] + off
    m = pats[e["entity_id"]].search(hyp) if e["entity_id"] in pats else None
    if m:
        w0 = len(hyp[:m.start()].rstrip().rsplit(" ", 6)[0]) if hyp[:m.start()].strip() else 0
        window = hyp[w0:m.end() + 4 * len(e["entity"]) + 30]
        c, o = best_rendering(window, e["entity"], e["form"], e["possessive"], same, longest)
        if c and c not in reject.get(e["entity_id"], ()):
            return c, "search", w0 + o
    return "", "", -1


def strict_key(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip().strip(EDGE).strip()


def score_model(model, clips, ents_by_sid, emap, reject, normalizer, rmap=None, rreject=None):
    pats = variant_patterns(emap)
    skeys = {k: {strict_key(v) for v in x["variants"]} for k, x in emap.items()}
    longest = {k: max([len(v) for v in x["variants"]] + [len(v) for v in (rmap or {}).get(k, {}).get("variants", {})], default=0)
               for k, x in emap.items()}
    rkeys = {k: {RK.variant_key(v) for v in x["variants"]} for k, x in (rmap or {}).items()}
    rpats = variant_patterns({k: {"variants": list(emap.get(k, {}).get("variants", [])) + list(x["variants"])}
                              for k, x in (rmap or {}).items()})
    rreject = {k: reject.get(k, set()) | (rreject or {}).get(k, set()) for k in set(reject) | set(rreject or {})}
    inst, rows = [], []
    for r in clips.itertuples(index=False):
        hyp = clean(r.hyp).lower()
        ref = r.reference
        g = ents_by_sid.get(r.sentence_id, [])
        ranges = entity_ranges(ref, [e["entity"] for e in g])
        sp = spans_for(ref, hyp, ranges) if hyp else [None] * len(g)
        repl = []
        for e, s in zip(g, sp):
            sk = skeys.get(e["entity_id"], set())

            def ssame(c, ent, form, sk=sk):
                return strict_key(c) in sk
            rend, how, start = find_entity(e, s, hyp, ssame, pats, reject, longest.get(e["entity_id"], 0))
            ok = bool(rend)
            if ok or e["form"] not in RK.RELAXED_FORMS:
                rrend, rhow = rend, how
            else:
                keys = rkeys.get(e["entity_id"], set())

                def rsame(c, ent, form, keys=keys, sk=sk):
                    return strict_key(c) in sk or RK.variant_key(c) in keys
                rrend, rhow, _ = find_entity(e, s, hyp, rsame, rpats, rreject, longest.get(e["entity_id"], 0))
            inst.append((model, e["entity_id"], r.sentence_id, r.accent, ok, how, rend,
                         bool(rrend), rhow, rrend))
            if ok and start >= 0:
                repl.append((start, start + len(rend), e["entity"].lower()))
        norm_hyp, last = hyp, len(hyp) + 1
        for a, b, canon in sorted(repl, reverse=True):
            if b <= last:
                norm_hyp = norm_hyp[:a] + canon + norm_hyp[b:]
                last = a
        ref_n = normalizer(ref)
        hyp_n = normalizer(norm_hyp)
        hyp_hf = normalizer(hyp)
        w = jiwer.process_words(ref_n, hyp_n) if ref_n else None
        c = jiwer.process_characters(ref_n, hyp_n) if ref_n else None
        whf = jiwer.process_words(ref_n, hyp_hf) if ref_n else None
        n_ref_w = len(ref_n.split())
        n_ref_c = len(ref_n)
        we = w.substitutions + w.deletions + w.insertions
        ce = c.substitutions + c.deletions + c.insertions
        we_hf = whf.substitutions + whf.deletions + whf.insertions
        n_ok = sum(1 for x in inst[-len(g):] if x[4]) if g else 0
        n_rok = sum(1 for x in inst[-len(g):] if x[7]) if g else 0
        rows.append({"model": model, "sentence_id": r.sentence_id, "accent": r.accent,
                     "entities": len(g), "entities_correct": n_ok, "all_entities_correct": n_ok == len(g),
                     "entities_correct_relaxed": n_rok, "all_entities_correct_relaxed": n_rok == len(g),
                     "ref_words": n_ref_w, "word_errors": we, "wer": we / max(n_ref_w, 1),
                     "substitutions": w.substitutions, "deletions": w.deletions, "insertions": w.insertions,
                     "ref_chars": n_ref_c, "char_errors": ce, "cer": ce / max(n_ref_c, 1),
                     "word_errors_hf_only": we_hf, "wer_hf_only": we_hf / max(n_ref_w, 1),
                     "reference_norm": ref_n, "prediction_norm": hyp_n, "prediction": r.hyp})
    return inst, rows


def _run(model, pred_path, clips_meta, ents_by_sid, emap, reject, rmap=None, rreject=None, cache=None):
    d = pd.read_csv(pred_path, usecols=["id", model], low_memory=False).rename(columns={model: "hyp"})
    d = d.merge(clips_meta[["id", "sentence_id", "accent", "reference"]], on="id")
    return score_model(model, d[["sentence_id", "accent", "reference", "hyp"]], ents_by_sid, emap, reject,
                       EnglishTextNormalizer(), rmap, rreject)


def summarise(rows, inst, by):
    r = rows.groupby(by).agg(clips=("sentence_id", "size"), success_rate=("all_entities_correct", "mean"),
                             success_rate_relaxed=("all_entities_correct_relaxed", "mean"),
                             ref_words=("ref_words", "sum"), word_errors=("word_errors", "sum"),
                             ref_chars=("ref_chars", "sum"), char_errors=("char_errors", "sum"))
    r["wer"] = r.word_errors / r.ref_words
    r["cer"] = r.char_errors / r.ref_chars
    e = inst.groupby(by).agg(entities=("correct", "size"), entity_accuracy=("correct", "mean"),
                             entity_accuracy_relaxed=("correct_relaxed", "mean"))
    t = e.join(r, how="left")
    for c in ("entity_accuracy", "success_rate", "entity_accuracy_relaxed", "success_rate_relaxed", "wer", "cer"):
        t[c] = (t[c] * 100).round(2)
    return t[["entities", "entity_accuracy", "entity_accuracy_relaxed", "clips", "success_rate",
              "success_rate_relaxed", "wer", "cer"]]


def load_benchmark(path):
    clips = pd.read_csv(path).rename(columns={"text": "reference"})
    ents = pd.DataFrame([(r.sentence_id, x["text"], x["type"]) for r in clips.drop_duplicates("sentence_id").itertuples()
                         for x in json.loads(r.entities)], columns=["sentence_id", "entity", "form"])
    emap = json.load(open(HERE / "data" / "entity_map_strict.json"))
    rmap = json.load(open(HERE / "data" / "entity_map_relaxed.json"))
    ids = {(x["sentence_id"], x["entity"]): k for k, x in emap.items()}
    ents["entity_id"] = [ids[(s, e)] for s, e in zip(ents.sentence_id, ents.entity)]
    ents["tier"] = [emap[k]["tier"] for k in ents.entity_id]
    ents["domain"] = [emap[k]["domain"] for k in ents.entity_id]
    ref = clips.drop_duplicates("sentence_id").set_index("sentence_id").reference
    ents["possessive"] = [bool(re.search(re.escape(e) + r"['’]s\b", ref.get(s, ""))) for e, s in zip(ents.entity, ents.sentence_id)]
    return clips, ents, emap, rmap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="the benchmark CSV from Hugging Face (id, sentence_id, accent, text, entities ...)")
    ap.add_argument("--preds", required=True, help="CSV with column `id` and one transcript column per model")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    clips, ents, emap, rmap = load_benchmark(args.benchmark)
    ents_by_sid = {s: g.to_dict("records") for s, g in ents.groupby("sentence_id", sort=False)}
    models = args.models or [c for c in pd.read_csv(args.preds, nrows=0).columns if c != "id"]
    print(f"{len(models)} models | {len(clips)} clips | {len(ents)} entities")
    job = partial(_run, pred_path=args.preds, clips_meta=clips, ents_by_sid=ents_by_sid, emap=emap, reject={},
                  rmap=rmap, rreject={})
    inst, rows = [], []
    with Pool(args.workers) as p:
        for k, (i, r) in enumerate(p.imap_unordered(job, models), 1):
            inst += i
            rows += r
            print(f"  scored {k}/{len(models)}", end="\r", flush=True)
    print()
    inst = pd.DataFrame(inst, columns=["model", "entity_id", "sentence_id", "accent", "correct", "found_by", "rendering",
                                       "correct_relaxed", "found_by_relaxed", "rendering_relaxed"])
    inst = inst.merge(ents[["entity_id", "entity", "form", "tier"]], on="entity_id").rename(columns={"form": "type"})
    meta = clips[["sentence_id", "accent", "id", "domain", "gender", "difficulty", "speaker_id", "duration_s"]]
    inst = inst.merge(meta, on=["sentence_id", "accent"])
    rows = pd.DataFrame(rows).merge(meta, on=["sentence_id", "accent"], how="left")
    lb = summarise(rows, inst, "model").sort_values(["entity_accuracy", "success_rate"], ascending=False)
    lb.insert(0, "rank", range(1, len(lb) + 1))
    lb.to_csv(out / "leaderboard.csv")
    for col in ("accent", "domain", "gender", "difficulty"):
        summarise(rows, inst, ["model", col]).to_csv(out / f"by_{col}.csv")
    for col in ("type", "tier"):
        t = inst.groupby(["model", col]).agg(entities=("correct", "size"), entity_accuracy=("correct", "mean"),
                                             entity_accuracy_relaxed=("correct_relaxed", "mean"))
        t[["entity_accuracy", "entity_accuracy_relaxed"]] = (t[["entity_accuracy", "entity_accuracy_relaxed"]] * 100).round(2)
        t.to_csv(out / f"by_{col}.csv")
    rows.drop(columns=["reference_norm", "prediction_norm"], errors="ignore").to_csv(out / "per_row.csv", index=False)
    inst.to_csv(out / "per_entity.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(lb.to_string())
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
