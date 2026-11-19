"""
MDM Material Intelligence Suite (All-in-One: Build + Search + Anomaly Detection + Web Dashboard)
Works on 00_all_cleaned_sorted_by_corporate_mdm.csv (~371,888 rows -> 166,739 unique codes).

Install dependencies once:
  pip install pandas scikit-learn fastapi uvicorn

Commands:
  1) Build unique master, NLP index (min_df=1, Short+Long only), & anomaly-tagged similarity report:
     python mdm_suite.py build "C:\\Users\\jays3\\Downloads\\mdm_cleaned_output\\00_all_cleaned_sorted_by_corporate_mdm.csv"

  2) Launch local web dashboard (http://127.0.0.1:8050):
     python mdm_suite.py dashboard

  3) Optional CLI search:
     python mdm_suite.py search --query "SKF 6205 2RS BEARING" --must "6205"
"""

import argparse
from collections import defaultdict
import io
from pathlib import Path
import re
import time
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


PN_PATTERN = re.compile(
    r"(?:PART\s*NO|P/N|PN|CAT|MODEL(?:/MACHINE\s*NO)?|EQUIP/TAG\s*MODEL|M|E)\s*[:\-]\s*([A-Z0-9][A-Z0-9\.\-\/]{2,})"
)


def clean_text_for_nlp(text: str) -> str:
    t = str(text).upper()
    # Normalize colons/semicolons/commas/brackets to spaces so SIZE:3 IN == SIZE: 3 IN
    t = re.sub(r"[:;,()\[\]]+", " ", t)
    t = re.sub(r"[^A-Z0-9\.\-\/]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_spec_tokens(text: str) -> set[str]:
    cleaned = clean_text_for_nlp(text)
    return {t for t in cleaned.split() if any(ch.isdigit() for ch in t)}


def extract_explicit_pns(text: str) -> set[str]:
    up = str(text).upper()
    found = set()
    for m in PN_PATTERN.findall(up):
        tok = m.strip(" .-/")
        if len(tok) >= 3 and any(ch.isdigit() for ch in tok) and tok not in {"NA", "NIL", "NONE"}:
            found.add(tok)
    return found


def build_unique_master(input_csv: Path) -> pd.DataFrame:
    print(f"[*] Loading cleaned & sorted file: {input_csv}")
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    print(f"[*] Loaded {len(df):,} rows.")

    if "clean_mat_code" not in df.columns:
        df["clean_mat_code"] = df["Corporate MDM Number"].astype(str).str.strip().str.upper()

    # Score completeness by Short & Long Description only when picking 1 row per Corporate MDM Number
    df["_score"] = (
        (df["Long Description"].str.len() > 0).astype(int) * 3
        + (df["Short Description"].str.len() > 0).astype(int) * 2
    )
    sort_cols = ["clean_mat_code", "_score"] + (["Edited Date"] if "Edited Date" in df.columns else [])
    asc_flags = [True, False] + ([False] if "Edited Date" in df.columns else [])
    df = df.sort_values(sort_cols, ascending=asc_flags, kind="mergesort")

    plants_agg = (
        df.loc[df["Plant"] != "", ["clean_mat_code", "Plant"]]
        .drop_duplicates()
        .groupby("clean_mat_code", sort=False)["Plant"]
        .agg(" | ".join)
    )
    sap_agg = (
        df.loc[df["SAP No"] != "", ["clean_mat_code", "SAP No"]]
        .drop_duplicates()
        .groupby("clean_mat_code", sort=False)["SAP No"]
        .agg(" | ".join)
    )

    unique_df = df.drop_duplicates(subset=["clean_mat_code"], keep="first").drop(columns=["_score"]).copy()
    unique_df["all_plants"] = unique_df["clean_mat_code"].map(plants_agg).fillna("")
    unique_df["all_sap_nos"] = unique_df["clean_mat_code"].map(sap_agg).fillna("")
    unique_df = unique_df.reset_index(drop=True)

    # 100% Short Description (2x) + Long Description (excludes Class)
    short_col = unique_df["Short Description"].map(clean_text_for_nlp)
    long_col = unique_df["Long Description"].map(clean_text_for_nlp)
    unique_df["_nlp_text"] = (short_col + " " + short_col + " " + long_col).str.strip()

    print(f"[✓] Extracted {len(unique_df):,} total unique Corporate MDM Numbers (1 row each).")
    return unique_df


def run_build(
    input_csv: Path,
    output_dir: Path,
    min_similarity: float = 0.75,
    top_k_per_item: int = 5,
    batch_size: int = 2000,
) -> None:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_df = build_unique_master(input_csv)

    master_out = output_dir / "unique_corporate_mdm_master.csv"
    unique_df.drop(columns=["_nlp_text"]).to_csv(master_out, index=False, encoding="utf-8-sig")
    print(f"[✓] Saved unique master: {master_out}")

    print("[*] Building NLP TF-IDF vector indexes (Short, Long, Combined; min_df=1)...")
    vec_kwargs = dict(
        lowercase=False,
        token_pattern=r"[A-Z0-9\.\-\/]{2,}",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer = TfidfVectorizer(**vec_kwargs)
    tfidf_matrix = normalize(vectorizer.fit_transform(unique_df["_nlp_text"]), norm="l2", copy=False)

    vec_short = TfidfVectorizer(**vec_kwargs)
    tfidf_short = normalize(vec_short.fit_transform(unique_df["Short Description"].map(clean_text_for_nlp)), norm="l2", copy=False)

    vec_long = TfidfVectorizer(**vec_kwargs)
    tfidf_long = normalize(vec_long.fit_transform(unique_df["Long Description"].map(clean_text_for_nlp)), norm="l2", copy=False)
    print(f"[*] TF-IDF matrix shape: {tfidf_matrix.shape[0]:,} codes x {tfidf_matrix.shape[1]:,} features.")

    index_path = output_dir / "mdm_nlp_index.joblib"
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "unique_df": unique_df.drop(columns=["_nlp_text"]),
        },
        index_path,
        compress=3,
    )
    print(f"[✓] Saved searchable NLP index: {index_path}")

    n_items = tfidf_matrix.shape[0]
    codes = unique_df["clean_mat_code"].to_numpy()
    short_descs = unique_df["Short Description"].to_numpy()
    long_descs = unique_df["Long Description"].to_numpy()
    uoms = unique_df["UOM"].to_numpy() if "UOM" in unique_df.columns else np.full(n_items, "")
    classes = unique_df["Class"].to_numpy() if "Class" in unique_df.columns else np.full(n_items, "")

    # 1) Cosine similarity >= threshold
    print(f"[*] Scanning all {n_items:,} unique codes for >= {min_similarity * 100:.0f}% similarity...")
    tfidf_t = tfidf_matrix.T.tocsr()
    pair_map: dict[tuple[int, int], float] = {}

    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        sim_block = (tfidf_matrix[start:end] @ tfidf_t).tocsr()

        for local_r in range(end - start):
            i = start + local_r
            row_start, row_end = sim_block.indptr[local_r], sim_block.indptr[local_r + 1]
            cols = sim_block.indices[row_start:row_end]
            vals = sim_block.data[row_start:row_end]

            mask = (cols > i) & (vals >= min_similarity)
            if not np.any(mask):
                continue

            cand_cols = cols[mask]
            cand_vals = vals[mask]
            if len(cand_vals) > top_k_per_item:
                top_idx = np.argpartition(cand_vals, -top_k_per_item)[-top_k_per_item:]
                cand_cols = cand_cols[top_idx]
                cand_vals = cand_vals[top_idx]

            for j, sim in zip(cand_cols, cand_vals):
                pair_map[(i, int(j))] = float(sim)

        if (end % 20000 == 0) or end == n_items:
            print(f"    ...scanned {end:,}/{n_items:,} codes ({len(pair_map):,} NLP pairs found)")

    # 2) Inverted index on explicit Part/Model Numbers to catch same-PN pairs even below 75% cosine
    print("[*] Indexing explicit Part/Model Numbers to catch subset/missing-characteristic pairs...")
    pn_sets: list[set[str]] = []
    pn_inverted: dict[str, list[int]] = defaultdict(list)
    for idx, (s, l) in enumerate(zip(short_descs, long_descs)):
        pns = extract_explicit_pns(f"{s} {l}")
        pn_sets.append(pns)
        for pn in pns:
            pn_inverted[pn].append(idx)

    pn_added = 0
    for pn, idxs in pn_inverted.items():
        if 2 <= len(idxs) <= 25:
            for a_pos in range(len(idxs)):
                for b_pos in range(a_pos + 1, len(idxs)):
                    i, j = idxs[a_pos], idxs[b_pos]
                    if i > j:
                        i, j = j, i
                    if (i, j) not in pair_map:
                        sim_val = float(tfidf_matrix[i].multiply(tfidf_matrix[j]).sum())
                        pair_map[(i, j)] = sim_val
                        pn_added += 1
    print(f"[*] Added {pn_added:,} extra same-Part-Number pairs with differing characteristic counts.")

    if not pair_map:
        sim_df = pd.DataFrame()
    else:
        pairs = list(pair_map.items())
        i_idx = np.array([k[0] for k, _ in pairs], dtype=np.int32)
        j_idx = np.array([k[1] for k, _ in pairs], dtype=np.int32)
        comb_pct = np.round(np.array([v for _, v in pairs], dtype=np.float32) * 100.0, 2)

        print("[*] Computing Short %, Long %, Token-Set Overlap %, Containment %, Characteristic Delta & Anomaly Tags...")
        short_pct = np.round(np.asarray(tfidf_short[i_idx].multiply(tfidf_short[j_idx]).sum(axis=1)).ravel() * 100.0, 2)
        long_pct = np.round(np.asarray(tfidf_long[i_idx].multiply(tfidf_long[j_idx]).sum(axis=1)).ravel() * 100.0, 2)

        token_sets = [set(clean_text_for_nlp(f"{s} {l}").split()) for s, l in zip(short_descs, long_descs)]
        spec_sets = [{t for t in ts if any(ch.isdigit() for ch in t)} for ts in token_sets]

        overlap_list, contain_list, delta_cnt_list = [], [], []
        extra1_list, extra2_list, shared_pn_list, spec_diff_list, anomaly_list = [], [], [], [], []

        for ii, jj in zip(i_idx, j_idx):
            t1, t2 = token_sets[ii], token_sets[jj]
            inter = len(t1 & t2)
            union = len(t1 | t2)
            min_len = min(len(t1), len(t2))

            ov = round((inter / union) * 100.0, 2) if union else 0.0
            ct = round((inter / min_len) * 100.0, 2) if min_len else 0.0
            diff1 = sorted(t1 - t2)
            diff2 = sorted(t2 - t1)
            delta_cnt = len(diff1) + len(diff2)

            shared_pns = sorted(pn_sets[ii] & pn_sets[jj])
            shared_pn_str = " | ".join(shared_pns) if shared_pns else "NONE"

            only_s1 = sorted(spec_sets[ii] - spec_sets[jj])
            only_s2 = sorted(spec_sets[jj] - spec_sets[ii])
            if not only_s1 and not only_s2:
                s_diff = "EXACT_SPEC_MATCH"
            else:
                s_diff = (" ".join(only_s1[:4]) + " vs " + " ".join(only_s2[:4])).strip()

            if s_diff == "EXACT_SPEC_MATCH" and ov >= 98.0:
                atype = "TRUE_EXACT_DUPLICATE"
            elif s_diff == "EXACT_SPEC_MATCH" and ov >= 85.0:
                atype = "JUMBLED_PARAMETERS"
            elif shared_pns and ct >= 70.0 and delta_cnt > 0:
                atype = "SAME_PN_SUBSET_SPECS"
            elif s_diff != "EXACT_SPEC_MATCH":
                atype = "PART_NO_CONFLICT"
            else:
                atype = "HIGH_TEXT_SIMILARITY"

            overlap_list.append(ov)
            contain_list.append(ct)
            delta_cnt_list.append(delta_cnt)
            extra1_list.append(" ".join(diff1[:6]))
            extra2_list.append(" ".join(diff2[:6]))
            shared_pn_list.append(shared_pn_str)
            spec_diff_list.append(s_diff)
            anomaly_list.append(atype)

        sim_df = pd.DataFrame(
            {
                "corporate_mdm_1": codes[i_idx],
                "short_desc_1": short_descs[i_idx],
                "long_desc_1": [s[:250] for s in long_descs[i_idx]],
                "corporate_mdm_2": codes[j_idx],
                "short_desc_2": short_descs[j_idx],
                "long_desc_2": [s[:250] for s in long_descs[j_idx]],
                "anomaly_type": anomaly_list,
                "combined_similarity_pct": comb_pct,
                "short_text_similarity_pct": short_pct,
                "long_text_similarity_pct": long_pct,
                "token_set_overlap_pct": overlap_list,
                "containment_pct": contain_list,
                "char_delta_count": delta_cnt_list,
                "shared_part_nos": shared_pn_list,
                "spec_diff": spec_diff_list,
                "extra_in_1": extra1_list,
                "extra_in_2": extra2_list,
                "same_class": (classes[i_idx] == classes[j_idx]) & (classes[i_idx] != ""),
                "same_uom": (uoms[i_idx] == uoms[j_idx]) & (uoms[i_idx] != ""),
            }
        ).sort_values(["token_set_overlap_pct", "containment_pct", "combined_similarity_pct"], ascending=False, kind="mergesort")

    sim_out = output_dir / "nlp_similar_mat_codes.csv"
    sim_df.to_csv(sim_out, index=False, encoding="utf-8-sig")
    print(f"\n[✓] Total anomaly/similarity pairs saved: {len(sim_df):,}")
    if not sim_df.empty:
        print(sim_df["anomaly_type"].value_counts().to_string())
    print(f"[✓] Saved similarity report: {sim_out}")
    print(f"[✓] Total build time: {time.time() - t0:.1f}s")


def search_bundle(
    bundle: dict,
    q: str,
    must: str = "",
    plant: str = "",
    mat_class: str = "",
    uom: str = "",
    min_sim: float = 10.0,
    top_n: int = 50,
) -> pd.DataFrame:
    df = bundle["unique_df"]
    q_clean = clean_text_for_nlp(q)
    if not q_clean:
        return pd.DataFrame()

    q_vec = normalize(bundle["vectorizer"].transform([q_clean]), norm="l2", copy=False)
    scores = (bundle["tfidf_matrix"] @ q_vec.T).toarray().ravel() * 100.0

    mask = scores >= min_sim
    if plant:
        mask &= df["all_plants"].str.contains(plant, case=False, regex=False)
    if mat_class and "Class" in df.columns:
        mask &= df["Class"].str.contains(mat_class, case=False, regex=False)
    if uom and "UOM" in df.columns:
        mask &= df["UOM"].str.upper() == uom.strip().upper()
    if must:
        combined = (df["Short Description"] + " " + df["Long Description"]).str.upper()
        for tok in [t.strip().upper() for t in must.split(",") if t.strip()]:
            mask &= combined.str.contains(tok, regex=False)

    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return pd.DataFrame()

    top_idx = idx[np.argsort(scores[idx])[::-1][:top_n]]
    cols = [
        c
        for c in [
            "clean_mat_code",
            "all_sap_nos",
            "Old Material Number",
            "Class",
            "UOM",
            "all_plants",
            "Short Description",
            "Long Description",
            "Status",
        ]
        if c in df.columns
    ]
    res = df.iloc[top_idx][cols].copy()
    res.insert(0, "similarity_pct", np.round(scores[top_idx], 2))
    return res


def run_cli_search(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.index)
    print(f"[✓] Ready ({len(bundle['unique_df']):,} unique Corporate MDM Numbers indexed).")
    if args.query:
        res = search_bundle(bundle, args.query, args.must, args.plant, args.class_filter, args.uom, top_n=args.top)
        print("[!] No matches found." if res.empty else res.to_string(index=False, max_colwidth=55))
        return

    print("Interactive Search (type 'exit' to quit)")
    while True:
        q = input("\nSearch query > ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        must = input("Must-include tokens (comma-separated, or Enter) > ").strip()
        res = search_bundle(bundle, q, must, args.plant, args.class_filter, args.uom, top_n=args.top)
        if res.empty:
            print("[!] No matches found.")
        else:
            print(res[["similarity_pct", "clean_mat_code", "all_sap_nos", "UOM", "Short Description"]].to_string(index=False, max_colwidth=50))


# ==============================================================================
# FASTAPI DIGITAL DASHBOARD
# ==============================================================================
app = FastAPI(title="MDM Material Intelligence Dashboard")
STATE: dict = {"bundle": None, "sim_df": pd.DataFrame(), "conflicts_df": pd.DataFrame(), "same_plant_df": pd.DataFrame(), "missing_df": pd.DataFrame(), "kpis": {}}


BATCH_RULES = [
    ("Electrical", "Cable & Wire", r"\b(?:CABLE|WIRE|CONDUCTOR|CORD|XLPE|ARMOURED|GLAND|LUG)\b"),
    ("Electrical", "UPS & Battery", r"\b(?:UPS|BATTERY|CELL|INVERTER|CHARGER|SMPS|RECTIFIER)\b"),
    ("Electrical", "Switchgear & Protection", r"\b(?:MCCB|MCB|RCCB|ELCB|ACB|VCB|BREAKER|CONTACTOR|RELAY|FUSE|SWITCHGEAR|ISOLATOR|STARTER)\b"),
    ("Electrical", "Motor & Drive", r"\b(?:MOTOR|VFD|DRIVE|INDUCTION|STATOR|ROTOR|ARMATURE)\b"),
    ("Electrical", "Transformer & Power", r"\b(?:TRANSFORMER|CAPACITOR|REACTOR|BUSBAR|INSULATOR|SURGE)\b"),
    ("Electrical", "Lighting & Fixtures", r"\b(?:LAMP|LED|LIGHT|LUMINAIRE|FLOODLIGHT|BULB|TUBE|BALLAST|CHOKE)\b"),
    ("Electrical", "Panel, Socket & Switch", r"\b(?:PANEL|SOCKET|PLUG|SWITCH|TERMINAL|JUNCTION BOX|ENCLOSURE|MC4)\b"),
    ("Mechanical", "Bearing & Housing", r"\b(?:BEARING|PLUMMER|PILLOW BLOCK|SLEEVE|BUSH|BUSHING)\b"),
    ("Mechanical", "Pump & Impeller", r"\b(?:PUMP|IMPELLER|VOLUTE|CASING|SLURRY|DIAPHRAGM)\b"),
    ("Mechanical", "Valve & Actuator", r"\b(?:VALVE|ACTUATOR|GATE|GLOBE|BALL VALVE|BUTTERFLY|NRV|CHECK VALVE|PRV)\b"),
    ("Mechanical", "Pipe, Flange & Fitting", r"\b(?:PIPE|TUBE|FLANGE|ELBOW|TEE|REDUCER|UNION|NIPPLE|COUPLING|HOSE|FITTING)\b"),
    ("Mechanical", "Gasket, Seal & O-Ring", r"\b(?:GASKET|SEAL|O-RING|ORING|PACKING|GLAND PACKING|OIL SEAL)\b"),
    ("Mechanical", "Fasteners & Hardware", r"\b(?:BOLT|NUT|WASHER|SCREW|STUD|ANCHOR|RIVET|PIN|CIRCLIP)\b"),
    ("Mechanical", "Belt, Chain & Gearbox", r"\b(?:BELT|V-BELT|CHAIN|SPROCKET|PULLEY|GEAR|GEARBOX|SHAFT|COUPLER)\b"),
    ("Mechanical", "Filter & Strainer", r"\b(?:FILTER|STRAINER|CARTRIDGE|ELEMENT|SEPARATOR|SILENCER)\b"),
    ("Mechanical", "Fan, Blower & Compressor", r"\b(?:FAN|BLOWER|COMPRESSOR|CONVEYOR|ROLLER|IDLER|LINER|CHUTE)\b"),
    ("Instrumentation & Automation", "Sensor, Gauge & Transmitter", r"\b(?:SENSOR|TRANSMITTER|GAUGE|THERMOCOUPLE|RTD|FLOWMETER|LEVEL|PRESSURE|ENCODER|TRANSDUCER)\b"),
    ("Instrumentation & Automation", "PLC, IO & Control", r"\b(?:PLC|HMI|MODULE|CARD|SOLENOID|CONTROLLER|CALIBRATOR|ANALYZER)\b"),
    ("Safety & PPE", "PPE & Safety", r"\b(?:HELMET|GLOVE|SHOE|BOOT|GOGGLE|MASK|RESPIRATOR|HARNESS|SAFETY|LOCKOUT|LOTO|FIRE)\b"),
    ("Consumables & Chemical", "Lubricant, Paint & Chemical", r"\b(?:OIL|GREASE|LUBRICANT|PAINT|PRIMER|THINNER|ADHESIVE|EPOXY|CHEMICAL|ACID|SOLVENT|CLEANER)\b"),
    ("Consumables & Chemical", "Welding & Tools", r"\b(?:WELDING|ELECTRODE|GRINDING|CUTTING|DRILL|SPANNER|WRENCH|HAMMER|CUTTER)\b"),
]


def assign_batches(df: pd.DataFrame) -> pd.DataFrame:
    print("[*] Segregating catalog into Discipline & Sub-Type batches...")
    combined = (df["Short Description"] + " " + df["Long Description"]).str.upper()
    disc = pd.Series("General / Other", index=df.index)
    sub = pd.Series("Uncategorized", index=df.index)
    assigned = pd.Series(False, index=df.index)

    for d_name, s_name, pat in BATCH_RULES:
        m = (~assigned) & combined.str.contains(pat, regex=True)
        disc.loc[m] = d_name
        sub.loc[m] = s_name
        assigned |= m

    df["batch_discipline"] = disc
    df["batch_subtype"] = sub
    return df


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MDM Material Intelligence Dashboard</title>
  <style>
    :root { --bg:#0b1220; --panel:#121b2e; --panel-2:#18233a; --border:#24324f; --text:#e6edf7; --muted:#94a3b8; --accent:#38bdf8; --good:#22c55e; --warn:#f59e0b; --bad:#ef4444; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
    header { padding:18px 28px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; background:#0d1628; }
    header h1 { margin:0; font-size:19px; }
    header span { color:var(--muted); font-size:13px; }
    main { max-width:1440px; margin:0 auto; padding:20px 28px 40px; }
    .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:20px; }
    .kpi { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
    .kpi .label { font-size:12px; color:var(--muted); text-transform:uppercase; }
    .kpi .val { font-size:24px; font-weight:700; margin-top:6px; color:var(--accent); }
    .kpi.warn .val { color:var(--warn); } .kpi.good .val { color:var(--good); }
    .tabs { display:flex; gap:8px; margin-bottom:16px; border-bottom:1px solid var(--border); padding-bottom:10px; }
    .tab { background:var(--panel); color:var(--muted); border:1px solid var(--border); padding:9px 16px; border-radius:8px; cursor:pointer; font-weight:600; font-size:13px; }
    .tab.active { background:var(--accent); color:#071322; border-color:var(--accent); }
    .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px; }
    .controls { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; align-items:end; }
    label { display:block; font-size:11px; color:var(--muted); margin-bottom:4px; text-transform:uppercase; }
    input, select { width:100%; padding:9px 10px; border-radius:7px; border:1px solid var(--border); background:var(--panel-2); color:var(--text); font-size:13px; }
    button { padding:9px 14px; border-radius:7px; border:none; background:var(--accent); color:#071322; font-weight:700; cursor:pointer; font-size:13px; }
    button.secondary { background:var(--panel-2); color:var(--text); border:1px solid var(--border); }
    .table-wrap { overflow-x:auto; max-height:600px; overflow-y:auto; border:1px solid var(--border); border-radius:8px; }
    table { width:100%; border-collapse:collapse; font-size:12.5px; }
    th, td { padding:9px 11px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }
    th { background:#142038; position:sticky; top:0; z-index:2; color:var(--muted); font-size:11px; text-transform:uppercase; }
    tr:hover td { background:rgba(56,189,248,0.06); }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:700; font-size:11px; background:rgba(56,189,248,0.16); color:var(--accent); }
    .badge.high { background:rgba(34,197,94,0.16); color:var(--good); }
    .badge.warn { background:rgba(245,158,11,0.16); color:var(--warn); }
    .badge.bad { background:rgba(239,68,68,0.16); color:var(--bad); }
    .hidden { display:none; }
    .meta { font-size:12px; color:var(--muted); margin-bottom:10px; }
    .long-txt { max-width:340px; color:#cbd5e1; font-size:12px; }
  </style>
</head>
<body>
  <header style="flex-wrap:wrap;gap:12px;">
    <div><h1>MDM Material Intelligence Dashboard</h1><span>Short &amp; Long Text NLP &bull; Batch Segregation &bull; Universal Search</span></div>
    <div style="display:flex;gap:8px;align-items:center;flex:1;max-width:560px;">
      <input id="u-q" placeholder="Universal Search: MDM code, SAP No, Part No, Short/Long text, Plant..." onkeydown="if(event.key==='Enter') runUniversal()" />
      <button onclick="runUniversal()">Search All</button>
    </div>
    <span id="status-pill">Loading...</span>
  </header>
  <main>
    <section class="kpis" id="kpi-grid"></section>
    <div class="tabs" style="flex-wrap:wrap;">
      <button class="tab active" onclick="switchTab('search', this)">1. NLP Parameter Search (166K+ Codes)</button>
      <button class="tab" onclick="switchTab('similar', this)">2. Anomaly &amp; Similarity Explorer</button>
      <button class="tab" onclick="switchTab('batches', this)">3. Batch Segregation (Electrical / Mechanical)</button>
      <button class="tab" onclick="switchTab('audit', this)">4. Data Quality &amp; Duplicate Audit</button>
      <button class="tab hidden" id="tab-btn-universal" onclick="switchTab('universal', this)">Universal Search Results</button>
    </div>

    <section id="tab-search">
      <div class="card"><div class="controls">
        <div style="grid-column:span 2;"><label>Short / Long Text Query</label><input id="s-q" placeholder="e.g. SKF 6205 2RS BEARING or MCCB 250A 415V" onkeydown="if(event.key==='Enter') runSearch()" /></div>
        <div><label>Must-Include Tokens</label><input id="s-must" placeholder="e.g. 250A, 415V" onkeydown="if(event.key==='Enter') runSearch()" /></div>
        <div><label>Plant</label><input id="s-plant" placeholder="e.g. CLZS" /></div>
        <div><label>UOM</label><select id="s-uom"><option value="">All UOMs</option></select></div>
        <div><label>Min Sim %</label><input id="s-min" type="number" value="15" min="0" max="100" /></div>
        <div style="display:flex;gap:8px;"><button onclick="runSearch()" style="flex:1;">Search</button><button class="secondary" onclick="exportSearch()">CSV</button></div>
      </div></div>
      <div class="card"><div class="meta" id="s-meta">Enter any short or long text parameters above.</div><div class="table-wrap"><table id="s-table"><thead></thead><tbody></tbody></table></div></div>
    </section>

    <section id="tab-similar" class="hidden">
      <div class="card"><div class="controls">
        <div style="grid-column:span 2;"><label>Filter by Code, PN, or Description</label><input id="sim-q" placeholder="e.g. VALVE, 1205863, or MDM code" onkeydown="if(event.key==='Enter') loadSimilar()" /></div>
        <div><label>Anomaly Type</label><select id="sim-type" onchange="loadSimilar()">
          <option value="ALL">All Anomalies</option>
          <option value="TRUE_EXACT_DUPLICATE">True Exact Duplicate</option>
          <option value="JUMBLED_PARAMETERS">Jumbled Parameters</option>
          <option value="SAME_PN_SUBSET_SPECS">Same PN (More/Fewer Specs)</option>
          <option value="PART_NO_CONFLICT">Part No Conflict</option>
          <option value="HIGH_TEXT_SIMILARITY">High Text Similarity</option>
        </select></div>
        <div><label>Min Token Overlap %</label><input id="sim-ov" type="number" value="0" min="0" max="100" /></div>
        <div><label>Min Containment %</label><input id="sim-ct" type="number" value="0" min="0" max="100" /></div>
        <div><label>UOM Filter</label><select id="sim-uom"><option value="false">Any UOM</option><option value="true">Same UOM Only</option></select></div>
        <div style="display:flex;gap:8px;"><button onclick="simPage=0;loadSimilar()" style="flex:1;">Apply</button><button class="secondary" onclick="exportSimilar()">CSV</button></div>
      </div></div>
      <div class="card">
        <div class="meta" style="display:flex;justify-content:space-between;align-items:center;">
          <span id="sim-meta">Loading anomaly pairs...</span>
          <div style="display:flex;gap:6px;align-items:center;">
            <button class="secondary" onclick="if(simPage>0){simPage--;loadSimilar();}">&larr; Prev</button>
            <button class="secondary" onclick="simPage++;loadSimilar();">Next &rarr;</button>
          </div>
        </div>
        <div class="table-wrap"><table id="sim-table"><thead></thead><tbody></tbody></table></div>
      </div>
    </section>

        <section id="tab-batches" class="hidden">
      <div class="card">
        <div class="controls">
          <div><label>Discipline</label><select id="b-disc" onchange="onDiscChange()"><option value="ALL">All Disciplines</option></select></div>
          <div><label>Material Sub-Type</label><select id="b-sub" onchange="batchPage=0;loadBatches()"><option value="ALL">All Sub-Types</option></select></div>
          <div style="grid-column:span 2;"><label>Filter Inside Batch</label><input id="b-q" placeholder="Filter by code, short/long text, or plant..." onkeydown="if(event.key==='Enter'){batchPage=0;loadBatches();}" /></div>
          <div style="display:flex;gap:8px;"><button onclick="batchPage=0;loadBatches()" style="flex:1;">Load Batch</button><button class="secondary" onclick="exportBatches()">CSV</button></div>
        </div>
        <div id="batch-summary" style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;"></div>
      </div>
      <div class="card">
        <div class="meta" style="display:flex;justify-content:space-between;align-items:center;">
          <span id="b-meta">Select a Discipline / Sub-Type above.</span>
          <div style="display:flex;gap:6px;align-items:center;">
            <button class="secondary" onclick="if(batchPage>0){batchPage--;loadBatches();}">&larr; Prev</button>
            <button class="secondary" onclick="batchPage++;loadBatches();">Next &rarr;</button>
          </div>
        </div>
        <div class="table-wrap"><table id="b-table"><thead></thead><tbody></tbody></table></div>
      </div>
    </section>

    <section id="tab-universal" class="hidden">
      <div class="card">
        <div class="meta" style="display:flex;justify-content:space-between;align-items:center;">
          <span id="u-meta">Universal Search across Unique Master, Similarity Pairs, and Audit tables</span>
          <button class="secondary" onclick="exportUniversal()">Export Master Hits CSV</button>
        </div>
        <h3 style="margin:6px 0 10px;font-size:14px;color:var(--accent);">Unique Master Matches (Code, SAP No, Old Mat No, Plant, Text)</h3>
        <div class="table-wrap" style="max-height:360px;margin-bottom:18px;"><table id="u-master-table"><thead></thead><tbody></tbody></table></div>
        <h3 style="margin:6px 0 10px;font-size:14px;color:var(--warn);">Similarity &amp; Anomaly Pair Matches</h3>
        <div class="table-wrap" style="max-height:320px;"><table id="u-sim-table"><thead></thead><tbody></tbody></table></div>
      </div>
    </section>

    <section id="tab-audit" class="hidden">
      <div class="card"><div class="controls">
        <div><label>Audit Segment</label><select id="a-kind" onchange="loadAudit()"><option value="conflicts">Master-Data Conflicts (SAP / Desc / UOM)</option><option value="same_plant">Same-Plant Duplicate Rows</option><option value="missing">Missing Corporate MDM Number Rows</option></select></div>
        <div style="grid-column:span 2;"><label>Filter Text</label><input id="a-q" placeholder="Filter by code, plant, status, or description..." onkeydown="if(event.key==='Enter') loadAudit()" /></div>
        <div><button onclick="auditPage=0;loadAudit()" style="width:100%;">Load</button></div>
      </div></div>
      <div class="card">
        <div class="meta" style="display:flex;justify-content:space-between;align-items:center;">
          <span id="a-meta">Select an audit segment above.</span>
          <div style="display:flex;gap:6px;align-items:center;">
            <button class="secondary" onclick="if(auditPage>0){auditPage--;loadAudit();}">&larr; Prev</button>
            <button class="secondary" onclick="auditPage++;loadAudit();">Next &rarr;</button>
          </div>
        </div>
        <div class="table-wrap"><table id="a-table"><thead></thead><tbody></tbody></table></div>
      </div>
    </section>
  </main>

  <script>
    const fmt = n => Number(n || 0).toLocaleString('en-IN');
    function switchTab(name, btn) {
      document.querySelectorAll('section[id^="tab-"]').forEach(el => el.classList.add('hidden'));
      document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + name).classList.remove('hidden');
      btn.classList.add('active');
      if (name === 'similar' && !window._simLoaded) { loadSimilar(); window._simLoaded = true; }
      if (name === 'batches' && !window._batchesLoaded) { initBatches(); window._batchesLoaded = true; }
      if (name === 'audit' && !window._auditLoaded) { loadAudit(); window._auditLoaded = true; }
    }
    async function init() {
      const k = await (await fetch('/api/kpis')).json();
      document.getElementById('status-pill').textContent = `${fmt(k.unique_codes)} Unique Codes Indexed`;
      document.getElementById('kpi-grid').innerHTML = `
        <div class="kpi good"><div class="label">Unique Corporate MDM</div><div class="val">${fmt(k.unique_codes)}</div></div>
        <div class="kpi"><div class="label">Total Valid Rows</div><div class="val">${fmt(k.total_valid_rows)}</div></div>
        <div class="kpi"><div class="label">Similarity / Anomaly Pairs</div><div class="val">${fmt(k.similar_pairs)}</div></div>
        <div class="kpi warn"><div class="label">Same-Plant Dups</div><div class="val">${fmt(k.same_plant_dups)}</div></div>
        <div class="kpi warn"><div class="label">Master Conflicts</div><div class="val">${fmt(k.master_conflicts)}</div></div>
        <div class="kpi warn"><div class="label">Missing MDM Code</div><div class="val">${fmt(k.missing_codes)}</div></div>`;
      const uomSel = document.getElementById('s-uom');
      (k.uoms || []).forEach(u => { const o = document.createElement('option'); o.value = u; o.textContent = u; uomSel.appendChild(o); });
    }
    function renderTable(tableId, rows) {
      const thead = document.querySelector(`#${tableId} thead`), tbody = document.querySelector(`#${tableId} tbody`);
      if (!rows.length) { thead.innerHTML = ''; tbody.innerHTML = '<tr><td style="padding:16px;color:#94a3b8;">No matching rows found.</td></tr>'; return; }
      const cols = Object.keys(rows[0]);
      thead.innerHTML = '<tr>' + cols.map(c => `<th>${c.replace(/_/g, ' ')}</th>`).join('') + '</tr>';
      tbody.innerHTML = rows.map(r => '<tr>' + cols.map(c => {
        let v = r[c] ?? '';
        if (c.includes('_pct')) {
          const cls = Number(v) >= 90 ? 'high' : (Number(v) >= 75 ? '' : 'warn');
          return `<td><span class="badge ${cls}">${v}%</span></td>`;
        }
        if (c === 'anomaly_type') {
          const cls = v === 'TRUE_EXACT_DUPLICATE' ? 'high' : (v === 'PART_NO_CONFLICT' ? 'bad' : 'warn');
          return `<td><span class="badge ${cls}">${v}</span></td>`;
        }
        if (c === 'spec_diff') {
          const cls = v === 'EXACT_SPEC_MATCH' ? 'high' : 'warn';
          return `<td><span class="badge ${cls}">${v}</span></td>`;
        }
        if (c.toLowerCase().includes('long')) return `<td class="long-txt">${String(v).slice(0, 220)}</td>`;
        return `<td>${v}</td>`;
      }).join('') + '</tr>').join('');
    }
    async function runSearch() {
      const p = new URLSearchParams({ q: document.getElementById('s-q').value, must: document.getElementById('s-must').value, plant: document.getElementById('s-plant').value, uom: document.getElementById('s-uom').value, min_sim: document.getElementById('s-min').value || '10', top_n: '100' });
      const data = await (await fetch('/api/search?' + p.toString())).json();
      document.getElementById('s-meta').textContent = `Showing top ${fmt(data.count)} unique Corporate MDM Numbers`;
      renderTable('s-table', data.results);
    }
    function exportSearch() {
      const p = new URLSearchParams({ q: document.getElementById('s-q').value, must: document.getElementById('s-must').value, plant: document.getElementById('s-plant').value, uom: document.getElementById('s-uom').value, min_sim: document.getElementById('s-min').value || '10', top_n: '500' });
      window.location.href = '/api/search/export?' + p.toString();
    }
    let simPage = 0, auditPage = 0; const PAGE_SIZE = 150;
    function simParams(limit, offset) {
      return new URLSearchParams({ q: document.getElementById('sim-q').value, anomaly_type: document.getElementById('sim-type').value, min_ov: document.getElementById('sim-ov').value || '0', min_ct: document.getElementById('sim-ct').value || '0', same_uom: document.getElementById('sim-uom').value, limit: String(limit), offset: String(offset) });
    }
    async function loadSimilar() {
      const offset = simPage * PAGE_SIZE;
      const data = await (await fetch('/api/similar?' + simParams(PAGE_SIZE, offset).toString())).json();
      if (data.results.length === 0 && simPage > 0) { simPage--; return; }
      const end = offset + data.results.length;
      document.getElementById('sim-meta').textContent = `${fmt(data.total_matching)} matching pairs (showing ${data.total_matching ? offset + 1 : 0}-${fmt(end)})`;
      renderTable('sim-table', data.results);
    }
    function exportSimilar() {
      window.location.href = '/api/similar/export?' + simParams(100000, 0).toString();
    }
    async function loadAudit() {
      const kind = document.getElementById('a-kind').value, q = document.getElementById('a-q').value, offset = auditPage * PAGE_SIZE;
      const data = await (await fetch(`/api/audit/${kind}?q=${encodeURIComponent(q)}&limit=${PAGE_SIZE}&offset=${offset}`)).json();
      if (data.results.length === 0 && auditPage > 0) { auditPage--; return; }
      const end = offset + data.results.length;
      document.getElementById('a-meta').textContent = `${fmt(data.total)} total rows (showing ${data.total ? offset + 1 : 0}-${fmt(end)})`;
      renderTable('a-table', data.results);
    }

    let batchPage = 0, batchCounts = {};
    async function initBatches() {
      const data = await (await fetch('/api/batches/summary')).json();
      batchCounts = data.subtypes_by_discipline || {};
      const discSel = document.getElementById('b-disc');
      Object.entries(data.disciplines || {}).forEach(([d, cnt]) => {
        const o = document.createElement('option'); o.value = d; o.textContent = `${d} (${fmt(cnt)})`; discSel.appendChild(o);
      });
      onDiscChange();
    }
    function onDiscChange() {
      const d = document.getElementById('b-disc').value;
      const subSel = document.getElementById('b-sub');
      subSel.innerHTML = '<option value="ALL">All Sub-Types</option>';
      const chips = [];
      const list = d === 'ALL' ? Object.entries(batchCounts).flatMap(([dn, obj]) => Object.entries(obj).map(([s, c]) => [s, c, dn])) : Object.entries(batchCounts[d] || {}).map(([s, c]) => [s, c, d]);
      list.sort((a,b) => b[1]-a[1]).forEach(([s, c, dn]) => {
        const o = document.createElement('option'); o.value = s; o.textContent = `${s} (${fmt(c)})`; subSel.appendChild(o);
        chips.push(`<span class="badge" style="cursor:pointer;" onclick="selectBatchChip('${dn}','${s}')">${s}: ${fmt(c)}</span>`);
      });
      document.getElementById('batch-summary').innerHTML = chips.slice(0, 22).join('');
      batchPage = 0; loadBatches();
    }
    function selectBatchChip(d, s) {
      document.getElementById('b-disc').value = d;
      onDiscChange();
      document.getElementById('b-sub').value = s;
      batchPage = 0; loadBatches();
    }
    async function loadBatches() {
      const d = document.getElementById('b-disc').value, s = document.getElementById('b-sub').value, q = document.getElementById('b-q').value;
      const offset = batchPage * PAGE_SIZE;
      const p = new URLSearchParams({ discipline: d, subtype: s, q, limit: String(PAGE_SIZE), offset: String(offset) });
      const data = await (await fetch('/api/batches?' + p.toString())).json();
      if (data.results.length === 0 && batchPage > 0) { batchPage--; return; }
      const end = offset + data.results.length;
      document.getElementById('b-meta').textContent = `${fmt(data.total)} items in batch (showing ${data.total ? offset + 1 : 0}-${fmt(end)})`;
      renderTable('b-table', data.results);
    }
    function exportBatches() {
      const p = new URLSearchParams({ discipline: document.getElementById('b-disc').value, subtype: document.getElementById('b-sub').value, q: document.getElementById('b-q').value });
      window.location.href = '/api/batches/export?' + p.toString();
    }
    async function runUniversal() {
      const q = document.getElementById('u-q').value.trim();
      if (!q) return;
      const btn = document.getElementById('tab-btn-universal');
      btn.classList.remove('hidden');
      switchTab('universal', btn);
      const data = await (await fetch(`/api/universal?q=${encodeURIComponent(q)}`)).json();
      document.getElementById('u-meta').textContent = `Universal Search "${q}": ${fmt(data.master_total)} unique codes matched, ${fmt(data.sim_total)} similarity pairs matched`;
      renderTable('u-master-table', data.master_results);
      renderTable('u-sim-table', data.sim_results);
    }
    function exportUniversal() {
      const q = document.getElementById('u-q').value.trim();
      if (q) window.location.href = `/api/universal/export?q=${encodeURIComponent(q)}`;
    }
    init();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return DASHBOARD_HTML


@app.get("/api/kpis")
def api_kpis():
    return STATE["kpis"]


@app.get("/api/search")
def api_search(q: str = "", must: str = "", plant: str = "", mat_class: str = "", uom: str = "", min_sim: float = Query(10.0, ge=0.0, le=100.0), top_n: int = Query(50, ge=1, le=500)):
    res = search_bundle(STATE["bundle"], q, must, plant, mat_class, uom, min_sim, top_n)
    return {"count": len(res), "results": res.to_dict(orient="records")}


@app.get("/api/search/export")
def api_search_export(q: str = "", must: str = "", plant: str = "", mat_class: str = "", uom: str = "", min_sim: float = 10.0, top_n: int = 500):
    res = search_bundle(STATE["bundle"], q, must, plant, mat_class, uom, min_sim, top_n)
    buf = io.StringIO()
    res.to_csv(buf, index=False, encoding="utf-8-sig")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=mdm_search_results.csv"})


def filter_similar_df(q: str = "", anomaly_type: str = "ALL", min_ov: float = 0.0, min_ct: float = 0.0, same_uom: bool = False) -> pd.DataFrame:
    df = STATE["sim_df"]
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if anomaly_type != "ALL" and "anomaly_type" in df.columns:
        mask &= df["anomaly_type"] == anomaly_type
    if min_ov > 0 and "token_set_overlap_pct" in df.columns:
        mask &= df["token_set_overlap_pct"] >= min_ov
    if min_ct > 0 and "containment_pct" in df.columns:
        mask &= df["containment_pct"] >= min_ct
    if same_uom and "same_uom" in df.columns:
        mask &= df["same_uom"].astype(str).str.upper() == "TRUE"
    if q.strip():
        pat = q.strip()
        mask &= (
            df["corporate_mdm_1"].str.contains(pat, case=False, regex=False)
            | df["corporate_mdm_2"].str.contains(pat, case=False, regex=False)
            | df["short_desc_1"].str.contains(pat, case=False, regex=False)
            | df["short_desc_2"].str.contains(pat, case=False, regex=False)
            | df["long_desc_1"].str.contains(pat, case=False, regex=False)
            | df["long_desc_2"].str.contains(pat, case=False, regex=False)
        )
    return df.loc[mask]


@app.get("/api/similar")
def api_similar(q: str = "", anomaly_type: str = "ALL", min_ov: float = 0.0, min_ct: float = 0.0, same_uom: bool = False, limit: int = Query(150, ge=1, le=500), offset: int = Query(0, ge=0)):
    sub = filter_similar_df(q, anomaly_type, min_ov, min_ct, same_uom)
    return {"total_matching": int(len(sub)), "results": sub.iloc[offset:offset + limit].to_dict(orient="records")}


@app.get("/api/similar/export")
def api_similar_export(q: str = "", anomaly_type: str = "ALL", min_ov: float = 0.0, min_ct: float = 0.0, same_uom: bool = False):
    sub = filter_similar_df(q, anomaly_type, min_ov, min_ct, same_uom)
    buf = io.StringIO()
    sub.to_csv(buf, index=False, encoding="utf-8-sig")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=mdm_anomalies_filtered.csv"})


@app.get("/api/audit/{kind}")
def api_audit(kind: str, q: str = "", limit: int = 150, offset: int = 0):
    df = {"conflicts": STATE["conflicts_df"], "same_plant": STATE["same_plant_df"], "missing": STATE["missing_df"]}.get(kind, pd.DataFrame())
    if df.empty:
        return {"total": 0, "results": []}
    if q.strip():
        pat = q.strip()
        mask = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include="object").columns:
            mask |= df[col].str.contains(pat, case=False, regex=False)
        df = df.loc[mask]
    return {"total": int(len(df)), "results": df.iloc[offset:offset + limit].to_dict(orient="records")}



@app.get("/api/batches/summary")
def api_batches_summary():
    df = STATE["bundle"]["unique_df"]
    disc_counts = df["batch_discipline"].value_counts().to_dict()
    sub_by_disc: dict[str, dict[str, int]] = {}
    for (d, s), cnt in df.groupby(["batch_discipline", "batch_subtype"], sort=False).size().items():
        sub_by_disc.setdefault(d, {})[s] = int(cnt)
    return {"disciplines": {k: int(v) for k, v in disc_counts.items()}, "subtypes_by_discipline": sub_by_disc}


def filter_batches_df(discipline: str = "ALL", subtype: str = "ALL", q: str = "") -> pd.DataFrame:
    df = STATE["bundle"]["unique_df"]
    mask = pd.Series(True, index=df.index)
    if discipline != "ALL":
        mask &= df["batch_discipline"] == discipline
    if subtype != "ALL":
        mask &= df["batch_subtype"] == subtype
    if q.strip():
        pat = q.strip()
        mask &= (
            df["clean_mat_code"].str.contains(pat, case=False, regex=False)
            | df["all_sap_nos"].str.contains(pat, case=False, regex=False)
            | df["Short Description"].str.contains(pat, case=False, regex=False)
            | df["Long Description"].str.contains(pat, case=False, regex=False)
            | df["all_plants"].str.contains(pat, case=False, regex=False)
        )
    cols = [c for c in ["batch_discipline", "batch_subtype", "clean_mat_code", "all_sap_nos", "UOM", "Class", "all_plants", "Short Description", "Long Description"] if c in df.columns]
    return df.loc[mask, cols]


@app.get("/api/batches")
def api_batches(discipline: str = "ALL", subtype: str = "ALL", q: str = "", limit: int = Query(150, ge=1, le=500), offset: int = Query(0, ge=0)):
    sub = filter_batches_df(discipline, subtype, q)
    return {"total": int(len(sub)), "results": sub.iloc[offset:offset + limit].to_dict(orient="records")}


@app.get("/api/batches/export")
def api_batches_export(discipline: str = "ALL", subtype: str = "ALL", q: str = ""):
    sub = filter_batches_df(discipline, subtype, q)
    buf = io.StringIO()
    sub.to_csv(buf, index=False, encoding="utf-8-sig")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=mdm_batch_export.csv"})


def filter_universal_master(q: str) -> pd.DataFrame:
    df = STATE["bundle"]["unique_df"]
    tokens = [t.strip() for t in q.split() if t.strip()]
    mask = pd.Series(True, index=df.index)
    combined = (
        df["clean_mat_code"] + " " + df.get("all_sap_nos", "") + " " + df.get("Old Material Number", "")
        + " " + df.get("all_plants", "") + " " + df.get("Class", "") + " "
        + df["Short Description"] + " " + df["Long Description"]
    )
    for tok in tokens:
        mask &= combined.str.contains(tok, case=False, regex=False)
    cols = [c for c in ["clean_mat_code", "all_sap_nos", "Old Material Number", "batch_discipline", "batch_subtype", "UOM", "all_plants", "Short Description", "Long Description"] if c in df.columns]
    return df.loc[mask, cols]


@app.get("/api/universal")
def api_universal(q: str = ""):
    if not q.strip():
        return {"master_total": 0, "master_results": [], "sim_total": 0, "sim_results": []}
    m_df = filter_universal_master(q)
    s_df = filter_similar_df(q=q)
    return {
        "master_total": int(len(m_df)),
        "master_results": m_df.head(150).to_dict(orient="records"),
        "sim_total": int(len(s_df)),
        "sim_results": s_df.head(150).to_dict(orient="records"),
    }


@app.get("/api/universal/export")
def api_universal_export(q: str = ""):
    m_df = filter_universal_master(q)
    buf = io.StringIO()
    m_df.to_csv(buf, index=False, encoding="utf-8-sig")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=mdm_universal_search.csv"})


def run_dashboard(nlp_dir: Path, clean_dir: Path, port: int) -> None:
    index_path = nlp_dir / "mdm_nlp_index.joblib"
    if not index_path.exists():
        raise FileNotFoundError(f"Could not find {index_path}. Run `python mdm_suite.py build ...` first.")
    print(f"[*] Loading NLP index from {index_path}...")
    bundle = joblib.load(index_path)
    bundle["unique_df"] = assign_batches(bundle["unique_df"])
    STATE["bundle"] = bundle
    unique_df = bundle["unique_df"]

    sim_path = nlp_dir / "nlp_similar_mat_codes.csv"
    sim_df = pd.read_csv(sim_path, dtype=str, keep_default_na=False) if sim_path.exists() else pd.DataFrame()
    for col in ["combined_similarity_pct", "short_text_similarity_pct", "long_text_similarity_pct", "token_set_overlap_pct", "containment_pct"]:
        if col in sim_df.columns:
            sim_df[col] = pd.to_numeric(sim_df[col], errors="coerce").fillna(0.0)
    STATE["sim_df"] = sim_df

    summary_path = clean_dir / "05_duplicate_summary_by_corporate_mdm.csv"
    if summary_path.exists():
        s_df = pd.read_csv(summary_path, dtype=str, keep_default_na=False)
        for col in ["total_rows", "distinct_plants", "distinct_short_descs", "distinct_sap_nos", "distinct_uoms"]:
            if col in s_df.columns:
                s_df[col] = pd.to_numeric(s_df[col], errors="coerce").fillna(0).astype(int)
        cond = pd.Series(False, index=s_df.index)
        for c in ["distinct_sap_nos", "distinct_short_descs", "distinct_uoms"]:
            if c in s_df.columns:
                cond |= s_df[c] > 1
        STATE["conflicts_df"] = s_df.loc[cond].copy()

    sp_path = clean_dir / "02_duplicates_same_plant.csv"
    STATE["same_plant_df"] = pd.read_csv(sp_path, dtype=str, keep_default_na=False) if sp_path.exists() else pd.DataFrame()
    miss_path = clean_dir / "04_missing_corporate_mdm.csv"
    STATE["missing_df"] = pd.read_csv(miss_path, dtype=str, keep_default_na=False) if miss_path.exists() else pd.DataFrame()

    STATE["kpis"] = {
        "unique_codes": int(len(unique_df)),
        "total_valid_rows": int(pd.to_numeric(unique_df["code_occurrences"], errors="coerce").fillna(1).sum()) if "code_occurrences" in unique_df.columns else len(unique_df),
        "similar_pairs": int(len(sim_df)),
        "same_plant_dups": int(len(STATE["same_plant_df"])),
        "missing_codes": int(len(STATE["missing_df"])),
        "master_conflicts": int(len(STATE["conflicts_df"])),
        "uoms": sorted([x for x in unique_df["UOM"].unique() if x])[:150] if "UOM" in unique_df.columns else [],
    }
    print(f"[✓] Dashboard ready at http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDM Material Intelligence Suite (Build + Search + Dashboard)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("build", help="Build unique master, NLP index, and anomaly similarity CSV")
    p_b.add_argument("input_csv", type=Path, help="Path to 00_all_cleaned_sorted_by_corporate_mdm.csv")
    p_b.add_argument("-o", "--output-dir", type=Path, default=Path("mdm_nlp_output_v4"), help="Output directory")
    p_b.add_argument("--min-sim", type=float, default=0.75, help="Similarity threshold 0..1 (default: 0.75)")
    p_b.add_argument("--top-k", type=int, default=5, help="Max similar matches kept per item")
    p_b.add_argument("--batch-size", type=int, default=2000, help="Batch size for sparse similarity scan")

    p_s = sub.add_parser("search", help="CLI search across all unique Corporate MDM Numbers")
    p_s.add_argument("--index", type=Path, default=Path("mdm_nlp_output_v4/mdm_nlp_index.joblib"))
    p_s.add_argument("--query", default="")
    p_s.add_argument("--must", default="")
    p_s.add_argument("--plant", default="")
    p_s.add_argument("--class-filter", default="")
    p_s.add_argument("--uom", default="")
    p_s.add_argument("--top", type=int, default=20)

    p_d = sub.add_parser("dashboard", help="Launch local browser dashboard")
    p_d.add_argument("--nlp-dir", type=Path, default=Path("mdm_nlp_output_v4"))
    p_d.add_argument("--clean-dir", type=Path, default=Path("mdm_cleaned_output"))
    p_d.add_argument("--port", type=int, default=8050)

    args = parser.parse_args()
    if args.cmd == "build":
        run_build(args.input_csv, args.output_dir, args.min_sim, args.top_k, args.batch_size)
    elif args.cmd == "search":
        run_cli_search(args)
    elif args.cmd == "dashboard":
        run_dashboard(args.nlp_dir, args.clean_dir, args.port)