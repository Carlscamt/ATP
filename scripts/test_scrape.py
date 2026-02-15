"""Scrape AO 2024 with full stats, save, and report on the data."""
import logging
import sys
import os
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("report")

from scraper.atp_scraper import ATPScraper
import polars as pl
from pathlib import Path

# Delete old data
old_path = Path("data/raw/matches.parquet")
if old_path.exists():
    old_path.unlink()
    logger.info("Deleted old matches.parquet")

scraper = ATPScraper({"rate_limit_seconds": 1.5})

# Scrape AO 2024 with stats
logger.info("Scraping Australian Open 2024 with Hawkeye stats...")
matches = scraper.scrape_tournament_results("australian-open", "580", 2024)
logger.info(f"Found {len(matches)} matches")

# Only fetch stats for main draw (not qualifying) to save time
main_draw = [m for m in matches if "qualif" not in m["round"].lower()]
qualifying = [m for m in matches if "qualif" in m["round"].lower()]
logger.info(f"Main draw: {len(main_draw)}, Qualifying: {len(qualifying)}")

# Enrich main draw with stats
stats_count = 0
for i, m in enumerate(main_draw):
    m["tournament_name"] = "Australian Open"
    m["tier"] = "Grand Slam"
    m["date"] = "2024-01-14"
    m["location"] = "Melbourne, Australia"
    
    raw_id = m.get("raw_match_id", "")
    if raw_id:
        try:
            stats = scraper.scrape_match_stats(2024, "580", raw_id)
            if stats:
                m.update(stats)
                stats_count += 1
        except Exception as e:
            logger.debug(f"Stats failed for {raw_id}: {e}")
    
    if (i + 1) % 20 == 0:
        logger.info(f"  Progress: {i+1}/{len(main_draw)} ({stats_count} with stats)")

# Add qualifying without stats
for m in qualifying:
    m["tournament_name"] = "Australian Open"
    m["tier"] = "Grand Slam"
    m["date"] = "2024-01-14"
    m["location"] = "Melbourne, Australia"

all_matches = main_draw + qualifying
logger.info(f"Total: {len(all_matches)} matches, {stats_count} with stats")

# Save
df = pl.DataFrame(all_matches)
Path("data/raw").mkdir(parents=True, exist_ok=True)
df.write_parquet("data/raw/matches.parquet")
logger.info(f"Saved to data/raw/matches.parquet")

# ---- REPORT ----
print("\n" + "=" * 70)
print("  DATA REPORT: Australian Open 2024 (with Hawkeye stats)")
print("=" * 70)

print(f"\n1. OVERVIEW")
print(f"   Total matches: {len(df)}")
print(f"   Total columns: {len(df.columns)}")
print(f"   Matches with stats: {stats_count}/{len(main_draw)} main draw ({100*stats_count/max(len(main_draw),1):.0f}%)")

print(f"\n2. COLUMNS ({len(df.columns)})")
for col in df.columns:
    dtype = str(df[col].dtype)
    nulls = df[col].null_count()
    null_pct = 100 * nulls / len(df)
    print(f"   {col:45s} {dtype:12s} nulls: {nulls:3d} ({null_pct:.0f}%)")

# Stats columns
stat_cols = [c for c in df.columns if c.startswith("p1_") or c.startswith("p2_")]
print(f"\n3. STAT COLUMNS ({len(stat_cols)})")
for col in sorted(stat_cols):
    if col.startswith("p1_"):
        base = col[3:]
        p1_col = f"p1_{base}"
        p2_col = f"p2_{base}"
        if p1_col in df.columns and p2_col in df.columns:
            p1_vals = df[p1_col].drop_nulls()
            p2_vals = df[p2_col].drop_nulls()
            if len(p1_vals) > 0:
                print(f"   {base:35s}  p1: mean={p1_vals.mean():7.1f}  p2: mean={p2_vals.mean():7.1f}  coverage: {len(p1_vals)}/{len(df)}")

# 4. Round distribution
print(f"\n4. ROUND DISTRIBUTION")
for row in df.group_by("round").len().sort("len", descending=True).iter_rows(named=True):
    has_stats = len(df.filter((pl.col("round") == row["round"]) & pl.col("p1_aces").is_not_null())) if "p1_aces" in df.columns else 0
    print(f"   {row['round']:30s}  {row['len']:3d} matches  ({has_stats} with stats)")

# 5. Sample match with full stats
print(f"\n5. SAMPLE: AO 2024 Final")
final = df.filter(pl.col("round") == "Finals")
if len(final) > 0:
    row = final.row(0, named=True)
    print(f"   {row.get('player1_name', '?')} vs {row.get('player2_name', '?')}")
    print(f"   Score: {row.get('score', '?')}  Winner: {row.get('winner', '?')}")
    print(f"   Duration: {row.get('match_duration', '?')}")
    print()
    print(f"   {'Stat':35s}  {'P1':>8s}  {'P2':>8s}")
    print(f"   {'-'*35}  {'-'*8}  {'-'*8}")
    for base in ["aces", "double_faults", "serve_rating", "first_serve_pct", 
                  "first_serve_won_pct", "second_serve_won_pct",
                  "break_points_saved_pct", "bp_converted_pct",
                  "return_rating", "total_points_won_pct",
                  "service_points_won_pct", "return_points_won_pct",
                  "total_points_won", "total_points"]:
        p1_val = row.get(f"p1_{base}", None)
        p2_val = row.get(f"p2_{base}", None)
        if p1_val is not None:
            print(f"   {base:35s}  {p1_val:8.0f}  {p2_val:8.0f}")
