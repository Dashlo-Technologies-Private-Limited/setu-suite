# db_patch.py
import sqlite3
import random

DB_PATH = "setu_operations.db"

def patch_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("[*] Applying schema upgrades to setu_operations.db...")

    # 1. Extend or create stock_bins with plant & rack attributes
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stock_bins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mat_code TEXT NOT NULL,
        plant TEXT NOT NULL DEFAULT 'ZAWAR',
        rack_id TEXT NOT NULL DEFAULT 'R1',
        current_qty REAL NOT NULL DEFAULT 0.0,
        reserved_qty REAL NOT NULL DEFAULT 0.0,
        min_safety_stock REAL NOT NULL DEFAULT 5.0,
        unit_price_inr REAL NOT NULL DEFAULT 1500.0,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(mat_code, plant, rack_id)
    )
    """)

    # 2. Add Cycle Count Physical Audit Ledger
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stock_audit_discrepancies (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        plant TEXT NOT NULL,
        rack_id TEXT NOT NULL,
        mat_code TEXT NOT NULL,
        book_qty REAL NOT NULL,
        scanned_physical_qty REAL NOT NULL,
        variance_qty REAL NOT NULL,
        variance_val_inr REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'FLAGGED',
        auditor_id TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 3. Add Vendor Performance & OTIF Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS vendor_performance (
        vendor_code TEXT PRIMARY KEY,
        vendor_name TEXT NOT NULL,
        promised_lead_time_days REAL NOT NULL,
        actual_lead_time_days REAL NOT NULL,
        otif_score REAL NOT NULL,
        sigma_lead_time REAL NOT NULL
    )
    """)

    # 4. Populate sample vendors for OTIF dynamic buffer sizing
    vendors = [
        ("VEND-KSB-01", "KSB Pumps & Valves Ltd", 45.0, 52.0, 76.5, 8.4),
        ("VEND-LT-02", "L&T Electrical & Automation", 30.0, 31.0, 94.2, 2.1),
        ("VEND-ABB-03", "ABB India Drive Systems", 60.0, 68.0, 81.0, 6.2),
        ("VEND-MW-04", "MeanWell Industrial Power", 20.0, 20.5, 98.0, 0.8),
        ("VEND-FLS-05", "FLSmidth Minerals Automation", 40.0, 48.0, 78.0, 7.5)
    ]
    cur.executemany("""
    INSERT OR REPLACE INTO vendor_performance 
    (vendor_code, vendor_name, promised_lead_time_days, actual_lead_time_days, otif_score, sigma_lead_time)
    VALUES (?, ?, ?, ?, ?, ?)
    """, vendors)

    # 5. Populate Multi-Plant distributions if empty
    cur.execute("SELECT count(*) FROM stock_bins")
    count = cur.fetchone()[0]
    if count == 0:
        print("[*] Populating initial multi-plant stock records from material_master...")
        cur.execute("SELECT mat_code FROM material_master LIMIT 200")
        mat_codes = [row[0] for row in cur.fetchall()]
        plants = ["ZAWAR", "CLZS", "DSC", "RAM", "DEBARI"]
        racks = ["R1", "R2", "R3", "R4"]

        records = []
        for code in mat_codes:
            # Seed 2-3 plants per material
            chosen_plants = random.sample(plants, k=random.randint(2, 4))
            for pl in chosen_plants:
                rack = random.choice(racks)
                curr = float(random.randint(2, 45))
                res = float(random.randint(0, int(curr * 0.3)))
                ss = float(random.randint(4, 15))
                price = float(random.randint(850, 45000))
                records.append((code, pl, rack, curr, res, ss, price))

        cur.executemany("""
        INSERT OR IGNORE INTO stock_bins 
        (mat_code, plant, rack_id, current_qty, reserved_qty, min_safety_stock, unit_price_inr)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        print(f"[+] Seeded {len(records)} plant/rack bin allocations.")

    conn.commit()
    conn.close()
    print("[✓] Database upgrade completed successfully.")

if __name__ == "__main__":
    patch_database()