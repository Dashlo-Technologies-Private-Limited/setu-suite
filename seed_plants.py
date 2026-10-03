import sqlite3
import random

DB_PATH = "setu_operations.db"

def seed_plants():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    print("[*] Checking stock_bins table and indexes...")
    # Ensure unique constraint exists for ON CONFLICT handling
    try:
        c.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_stock_bins 
            ON stock_bins(sap_mat_code, plant, storage_location, rack_no)
        """)
    except Exception as e:
        print("[!] Index notice:", e)

    # Fetch materials from material_master
    c.execute("SELECT sap_mat_code FROM material_master LIMIT 400")
    rows = c.fetchall()

    if not rows:
        print("[!] Warning: material_master table appears empty. Ingest data first.")
        conn.close()
        return

    codes = [r[0] for r in rows]
    plants = ["ZAWAR", "CLZS", "DSC", "RAM", "DEBARI"]
    racks = ["R1", "R2", "R3", "R4"]

    print(f"[*] Allocating {len(codes)} materials across 5 Rajasthan plant facilities...")
    inserted_count = 0

    for code in codes:
        # Assign each code to 2-3 different plants
        chosen_plants = random.sample(plants, k=random.randint(2, 3))
        for pl in chosen_plants:
            rack = random.choice(racks)
            qty = random.randint(4, 45)
            c.execute("""
                INSERT INTO stock_bins (sap_mat_code, plant, storage_location, rack_no, qty)
                VALUES (?, ?, 'MAIN-WH', ?, ?)
                ON CONFLICT(sap_mat_code, plant, storage_location, rack_no)
                DO UPDATE SET qty = excluded.qty
            """, (code, pl, rack, qty))
            inserted_count += 1

    conn.commit()

    c.execute("SELECT COUNT(*) FROM stock_bins")
    total_bins = c.fetchone()[0]
    print(f"[✓] Success! Added/updated {inserted_count} plant records.")
    print(f"[✓] Total active rows in stock_bins: {total_bins}")

    # Check distribution per plant
    c.execute("SELECT plant, count(*) FROM stock_bins GROUP BY plant")
    for pl, cnt in c.fetchall():
        print(f"    - {pl}: {cnt} items")

    conn.close()

if __name__ == "__main__":
    seed_plants()