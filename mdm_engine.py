"""
MDM Material Intelligence Router for Project SETU
Loads mdm_nlp_output_v4/ and mdm_cleaned_output/ when present,
and falls back to setu_operations.db (material_master) if not yet built.
"""

import io
from pathlib import Path
import re
import sqlite3
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

mdm_router = APIRouter(prefix="/api/mdm", tags=["MDM Intelligence"])

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

_STATE: dict = {"loaded": False}


def clean_text_for_nlp(text: str) -> str:
    t = str(text).upper()
    t = re.sub(r"[:;,()\[\]]+", " ", t)
    t = re.sub(r"[^A-Z0-9\.\-\/]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def assign_batches(df: pd.DataFrame) -> pd.DataFrame:
    short_s = df["Short Description"].astype(str) if "Short Description" in df.columns else pd.Series("", index=df.index)
    long_s = df["Long Description"].astype(str) if "Long Description" in df.columns else pd.Series("", index=df.index)
    combined = (short_s + " " + long_s).str.upper()

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


def _ensure_loaded() -> None:
    if _STATE["loaded"]:
        return

    nlp_dir = Path("mdm_nlp_output_v4")
    clean_dir = Path("mdm_cleaned_output")
    index_path = nlp_dir / "mdm_nlp_index.joblib"

    if index_path.exists():
        bundle = joblib.load(index_path)
        bundle["unique_df"] = assign_batches(bundle["unique_df"])
        source_mode = "MDM_NLP_INDEX"
    else:
        # Fallback to setu_operations.db material_master so SETU always works out-of-the-box
        conn = sqlite3.connect("setu_operations.db")
        df = pd.read_sql_query(
            """
            SELECT m.sap_mat_code AS clean_mat_code,
                   m.sap_mat_code AS all_sap_nos,
                   m.oem_tag AS "Old Material Number",
                   m.category AS "Class",
                   '' AS "UOM",
                   m.short_text AS "Short Description",
                   m.long_text AS "Long Description",
                   m.criticality AS "Status",
                   COALESCE(GROUP_CONCAT(DISTINCT s.plant), '') AS all_plants
            FROM material_master m
            LEFT JOIN stock_bins s ON m.sap_mat_code = s.sap_mat_code
            GROUP BY m.sap_mat_code
            """,
            conn,
        ).fillna("")
        conn.close()
        df = assign_batches(df)
        nlp_text = (
            df["Short Description"].map(clean_text_for_nlp)
            + " "
            + df["Short Description"].map(clean_text_for_nlp)
            + " "
            + df["Long Description"].map(clean_text_for_nlp)
        ).str.strip()
        vec = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"[A-Z0-9\.\-\/]{2,}",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )
        mat = normalize(vec.fit_transform(nlp_text), norm="l2", copy=False) if len(df) else None
        bundle = {"vectorizer": vec, "tfidf_matrix": mat, "unique_df": df}
        source_mode = "SQLITE_FALLBACK"

    sim_path = nlp_dir / "nlp_similar_mat_codes.csv"
    sim_df = pd.read_csv(sim_path, dtype=str, keep_default_na=False) if sim_path.exists() else pd.DataFrame()
    for col in ["combined_similarity_pct", "short_text_similarity_pct", "long_text_similarity_pct", "token_set_overlap_pct", "containment_pct"]:
        if col in sim_df.columns:
            sim_df[col] = pd.to_numeric(sim_df[col], errors="coerce").fillna(0.0)

    conflicts_df = pd.DataFrame()
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
        conflicts_df = s_df.loc[cond].copy()

    sp_path = clean_dir / "02_duplicates_same_plant.csv"
    same_plant_df = pd.read_csv(sp_path, dtype=str, keep_default_na=False) if sp_path.exists() else pd.DataFrame()
    miss_path = clean_dir / "04_missing_corporate_mdm.csv"
    missing_df = pd.read_csv(miss_path, dtype=str, keep_default_na=False) if miss_path.exists() else pd.DataFrame()

    unique_df = bundle["unique_df"]
    total_valid_rows = (
        int(pd.to_numeric(unique_df["code_occurrences"], errors="coerce").fillna(1).sum())
        if "code_occurrences" in unique_df.columns
        else len(unique_df)
    )

    _STATE.update(
        {
            "loaded": True,
            "source_mode": source_mode,
            "bundle": bundle,
            "sim_df": sim_df,
            "conflicts_df": conflicts_df,
            "same_plant_df": same_plant_df,
            "missing_df": missing_df,
            "kpis": {
                "source_mode": source_mode,
                "unique_codes": int(len(unique_df)),
                "total_valid_rows": total_valid_rows,
                "similar_pairs": int(len(sim_df)),
                "same_plant_dups": int(len(same_plant_df)),
                "missing_codes": int(len(missing_df)),
                "master_conflicts": int(len(conflicts_df)),
                "uoms": sorted([x for x in unique_df["UOM"].unique() if x])[:150] if "UOM" in unique_df.columns else [],
            },
        }
    )


def _search_df(q: str, must: str = "", plant: str = "", uom: str = "", min_sim: float = 10.0, top_n: int = 100) -> pd.DataFrame:
    _ensure_loaded()
    bundle = _STATE["bundle"]
    df = bundle["unique_df"]
    q_clean = clean_text_for_nlp(q)
    if not q_clean or bundle["tfidf_matrix"] is None:
        return pd.DataFrame()

    q_vec = normalize(bundle["vectorizer"].transform([q_clean]), norm="l2", copy=False)
    scores = (bundle["tfidf_matrix"] @ q_vec.T).toarray().ravel() * 100.0
    mask = scores >= min_sim
    if plant:
        mask &= df["all_plants"].str.contains(plant, case=False, regex=False)
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
    cols = [c for c in ["clean_mat_code", "all_sap_nos", "batch_discipline", "batch_subtype", "UOM", "all_plants", "Short Description", "Long Description"] if c in df.columns]
    res = df.iloc[top_idx][cols].copy()
    res.insert(0, "similarity_pct", np.round(scores[top_idx], 2))
    return res


def _filter_similar(q: str = "", anomaly_type: str = "ALL", min_ov: float = 0.0, min_ct: float = 0.0, same_uom: bool = False) -> pd.DataFrame:
    _ensure_loaded()
    df = _STATE["sim_df"]
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


def _filter_batches(discipline: str = "ALL", subtype: str = "ALL", q: str = "") -> pd.DataFrame:
    _ensure_loaded()
    df = _STATE["bundle"]["unique_df"]
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


def _filter_universal_master(q: str) -> pd.DataFrame:
    _ensure_loaded()
    df = _STATE["bundle"]["unique_df"]
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


@mdm_router.get("/kpis")
def mdm_kpis():
    _ensure_loaded()
    return _STATE["kpis"]


@mdm_router.get("/search")
def mdm_search(q: str = "", must: str = "", plant: str = "", uom: str = "", min_sim: float = Query(10.0, ge=0.0, le=100.0), top_n: int = Query(100, ge=1, le=500)):
    res = _search_df(q, must, plant, uom, min_sim, top_n)
    return {"count": len(res), "results": res.to_dict(orient="records")}


@mdm_router.get("/search/export")
def mdm_search_export(q: str = "", must: str = "", plant: str = "", uom: str = "", min_sim: float = 10.0):
    res = _search_df(q, must, plant, uom, min_sim, top_n=500)
    buf = io.StringIO()
    res.to_csv(buf, index=False, encoding="utf-8-sig")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=mdm_search_results.csv"})


@mdm_router.get("/similar")
def mdm_similar(q: str = "", anomaly_type: str = "ALL", min_ov: float = 0.0, min_ct: float = 0.0, same_uom: bool = False, limit: int = Query(150, ge=1, le=500), offset: int = Query(0, ge=0)):
    sub = _filter_similar(q, anomaly_type, min_ov, min_ct, same_uom)
    return {"total_matching": int(len(sub)), "results": sub.iloc[offset:offset + limit].to_dict(orient="records")}


@mdm_router.get("/similar/export")
def mdm_similar_export(q: str = "", anomaly_type: str = "ALL", min_ov: float = 0.0, min_ct: float = 0.0, same_uom: bool = False):
    sub = _filter_similar(q, anomaly_type, min_ov, min_ct, same_uom)
    buf = io.StringIO()
    sub.to_csv(buf, index=False, encoding="utf-8-sig")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=mdm_anomalies_filtered.csv"})


@mdm_router.get("/batches/summary")
def mdm_batches_summary():
    _ensure_loaded()
    df = _STATE["bundle"]["unique_df"]
    disc_counts = df["batch_discipline"].value_counts().to_dict()
    sub_by_disc: dict[str, dict[str, int]] = {}
    for (d, s), cnt in df.groupby(["batch_discipline", "batch_subtype"], sort=False).size().items():
        sub_by_disc.setdefault(d, {})[s] = int(cnt)
    return {"disciplines": {k: int(v) for k, v in disc_counts.items()}, "subtypes_by_discipline": sub_by_disc}


@mdm_router.get("/batches")
def mdm_batches(discipline: str = "ALL", subtype: str = "ALL", q: str = "", limit: int = Query(150, ge=1, le=500), offset: int = Query(0, ge=0)):
    sub = _filter_batches(discipline, subtype, q)
    return {"total": int(len(sub)), "results": sub.iloc[offset:offset + limit].to_dict(orient="records")}


@mdm_router.get("/batches/export")
def mdm_batches_export(discipline: str = "ALL", subtype: str = "ALL", q: str = ""):
    sub = _filter_batches(discipline, subtype, q)
    buf = io.StringIO()
    sub.to_csv(buf, index=False, encoding="utf-8-sig")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=mdm_batch_export.csv"})


@mdm_router.get("/universal")
def mdm_universal(q: str = ""):
    if not q.strip():
        return {"master_total": 0, "master_results": [], "sim_total": 0, "sim_results": []}
    m_df = _filter_universal_master(q)
    s_df = _filter_similar(q=q)
    return {
        "master_total": int(len(m_df)),
        "master_results": m_df.head(150).to_dict(orient="records"),
        "sim_total": int(len(s_df)),
        "sim_results": s_df.head(150).to_dict(orient="records"),
    }


@mdm_router.get("/universal/export")
def mdm_universal_export(q: str = ""):
    m_df = _filter_universal_master(q)
    buf = io.StringIO()
    m_df.to_csv(buf, index=False, encoding="utf-8-sig")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=mdm_universal_search.csv"})


@mdm_router.get("/audit/{kind}")
def mdm_audit(kind: str, q: str = "", limit: int = 150, offset: int = 0):
    _ensure_loaded()
    df = {"conflicts": _STATE["conflicts_df"], "same_plant": _STATE["same_plant_df"], "missing": _STATE["missing_df"]}.get(kind, pd.DataFrame())
    if df.empty:
        return {"total": 0, "results": []}
    if q.strip():
        pat = q.strip()
        mask = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include="object").columns:
            mask |= df[col].str.contains(pat, case=False, regex=False)
        df = df.loc[mask]
    return {"total": int(len(df)), "results": df.iloc[offset:offset + limit].to_dict(orient="records")}
