import os
import sqlite3
import pandas as pd
import numpy as np

EXCEL_FILE = "$$ MATERIAL CODES - 05.08.25 $$.XLSX"
DB_PATH = "setu_operations.db"

if not os.path.exists(EXCEL_FILE):
    candidates = [f for f in os.listdir('.') if "MATERIAL CODES" in f and f.endswith(('.xlsx', '.XLSX'))]
    if candidates:
        EXCEL_FILE = candidates[0]
    else:
        raise FileNotFoundError(f"Could not locate '{EXCEL_FILE}' in {os.getcwd()}")

print(f"[*] Reading all sheets from: {EXCEL_FILE}...")
xls = pd.ExcelFile(EXCEL_FILE)

all_materials_dict = {}

# 1. First priority: Extract from categorized engineering sheets (preserves specific category names)
category_sheets = ['MAT. CODE', 'MATERIAL CODES', 'WINDER', 'STATIONERY']
for sheet in category_sheets:
    if sheet not in xls.sheet_names:
        continue
    df_s = pd.read_excel(xls, sheet_name=sheet)
    for c_idx in range(len(df_s.columns)):
        col_header = str(df_s.iloc[0, c_idx]).strip().lower()
        if col_header in ['material', 'material code']:
            desc_col = c_idx + 1
            uom_col = c_idx + 2
            
            category = sheet
            for back_idx in range(c_idx, -1, -1):
                col_name = str(df_s.columns[back_idx]).strip()
                if not col_name.startswith('Unnamed'):
                    category = col_name
                    break
                    
            for r in range(1, len(df_s)):
                code = str(df_s.iloc[r, c_idx]).strip()
                desc = str(df_s.iloc[r, desc_col]).strip() if desc_col < len(df_s.columns) else ""
                uom = str(df_s.iloc[r, uom_col]).strip() if uom_col < len(df_s.columns) else "NO"
                if code and code != 'nan' and len(code) >= 6 and desc and desc != 'nan':
                    all_materials_dict[code] = {
                        'sap_mat_code': code,
                        'desc': desc,
                        'category': category,
                        'uom': uom,
                        'rop': 2.0,
                        'max_stock': 5.0
                    }

# 2. Second priority: Ingest all 5,924 items from '1070 MRP-2025'
if '1070 MRP-2025' in xls.sheet_names:
    df_mrp = pd.read_excel(xls, sheet_name='1070 MRP-2025')
    for _, row in df_mrp.iterrows():
        code = str(row['MATERIAL']).strip()
        if not code or code == 'nan' or len(code) < 6:
            continue
        desc = str(row['DESCRIPTION']).strip() if pd.notnull(row['DESCRIPTION']) else "HZL SPARE"
        uom = str(row['UOM']).strip() if pd.notnull(row['UOM']) else "NO"
        try:
            rop = float(row['REORDER POINT'])
        except:
            rop = 2.0
        try:
            max_stk = float(row['MAX. STOCK'])
        except:
            max_stk = 5.0
            
        if code in all_materials_dict:
            all_materials_dict[code]['rop'] = rop
            all_materials_dict[code]['max_stock'] = max_stk
        else:
            all_materials_dict[code] = {
                'sap_mat_code': code,
                'desc': desc,
                'category': 'General Spares & Consumables',
                'uom': uom,
                'rop': rop,
                'max_stock': max_stk
            }

print(f"[*] Aggregated {len(all_materials_dict)} total unique material codes.")

# 3. Setup SQLite Database Schema
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("DROP TABLE IF EXISTS material_master")
c.execute("DROP TABLE IF EXISTS stock_bins")

c.execute("""CREATE TABLE material_master (
    oem_tag TEXT PRIMARY KEY,
    sap_mat_code TEXT,
    short_text TEXT,
    long_text TEXT,
    specs TEXT,
    category TEXT,
    unit_cost REAL,
    lead_time_days INTEGER,
    criticality TEXT,
    aging_days INTEGER,
    shutdown_flag INTEGER DEFAULT 0
)""")

c.execute("""CREATE TABLE stock_bins (
    sap_mat_code TEXT,
    plant TEXT,
    storage_location TEXT,
    rack_no TEXT,
    qty REAL,
    PRIMARY KEY (sap_mat_code, plant, storage_location, rack_no)
)""")

# Index for instant search across all 6,000+ items
c.execute("CREATE INDEX IF NOT EXISTS idx_mat_code ON material_master(sap_mat_code)")
c.execute("CREATE INDEX IF NOT EXISTS idx_stock_code ON stock_bins(sap_mat_code)")

plants = ['Chanderiya', 'Dariba', 'Agucha', 'Zawar', 'Debari']
racks = ['R1', 'R2', 'R3', 'R4']

def classify_item(desc, cat):
    desc_u = desc.upper()
    if any(k in desc_u for k in ['PUMP', 'MCCB', 'STARTER', 'WINDER', 'BREAKER', 'SMPS', 'CONTACTOR', 'MOTOR', 'IMPELLER']):
        crit = 'Vital'
        cost = float(np.random.choice([14500, 22000, 35000, 52000, 78000]))
        lead_time = int(np.random.choice([45, 60, 90]))
    elif any(k in desc_u for k in ['RELAY', 'SWITCH', 'CABLE', 'VALVE', 'METER', 'TIMER', 'FUSE', 'REGULATOR', 'BEARING']):
        crit = 'Essential'
        cost = float(np.random.choice([2800, 4500, 6800, 11000]))
        lead_time = int(np.random.choice([30, 45]))
    else:
        crit = 'Desirable'
        cost = float(np.random.choice([450, 850, 1400, 2200]))
        lead_time = int(np.random.choice([15, 25]))
    return crit, cost, lead_time

mat_rows = []
bin_rows = []

for code, data in all_materials_dict.items():
    desc = data['desc']
    cat = data['category']
    uom = data['uom']
    rop = data['rop']
    
    crit, cost, lt = classify_item(desc, cat)
    aging = int(np.random.choice([20, 40, 75, 190, 410]))
    shutdown = 1 if crit == 'Vital' and np.random.rand() > 0.5 else 0
    specs = f"Cat: {cat} | UoM: {uom} | SAP Standard ROP: {rop}"
    long_desc = f"HZL Industrial Master: {desc}. Used across Rajasthan Mining and Smelting units."
    
    mat_rows.append((f"OEM-{code}", code, desc, long_desc, specs, cat, cost, lt, crit, aging, shutdown))
    
    # Realistic stock around ROP
    base_qty = max(0.0, float(int(np.random.normal(rop, max(1.0, rop * 0.4)))))
    assigned_plant = np.random.choice(plants)
    assigned_rack = np.random.choice(racks)
    
    bin_rows.append((code, assigned_plant, "MAIN-WH", assigned_rack, base_qty))
    
    # Inter-plant surplus distribution for cross-cluster balance testing
    if crit == 'Vital' and np.random.rand() > 0.65:
        alt_plant = 'Dariba' if assigned_plant == 'Chanderiya' else 'Chanderiya'
        bin_rows.append((code, alt_plant, "MAIN-WH", "R3", float(base_qty + 10)))

c.executemany("INSERT INTO material_master VALUES (?,?,?,?,?,?,?,?,?,?,?)", mat_rows)
c.executemany("INSERT OR REPLACE INTO stock_bins VALUES (?,?,?,?,?)", bin_rows)

conn.commit()
print(f"[SUCCESS] Ingested {len(mat_rows)} material codes and {len(bin_rows)} stock bin records into '{DB_PATH}'!")
conn.close()