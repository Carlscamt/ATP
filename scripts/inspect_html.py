"""Parse the Hawkeye JSON response and show all available stats."""
from curl_cffi import requests as cffi_requests
import json

url = "https://www.atptour.com/-/Hawkeye/MatchStats/2024/580/ms001"
resp = cffi_requests.get(url, impersonate="chrome")
data = resp.json()

# Save full JSON
with open("data/debug/hawkeye_full.json", "w") as f:
    json.dump(data, f, indent=2)

print("=" * 70)
print("  HAWKEYE API RESPONSE STRUCTURE")
print("=" * 70)

def show_structure(obj, prefix="", depth=0, max_depth=4):
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, dict):
                print(f"{prefix}{key}: {{dict with {len(val)} keys}}")
                show_structure(val, prefix + "  ", depth + 1, max_depth)
            elif isinstance(val, list):
                print(f"{prefix}{key}: [list of {len(val)} items]")
                if val and isinstance(val[0], dict):
                    show_structure(val[0], prefix + "  [0].", depth + 1, max_depth)
            else:
                val_str = str(val)[:80]
                print(f"{prefix}{key}: {val_str}")
    elif isinstance(obj, list) and obj:
        if isinstance(obj[0], dict):
            show_structure(obj[0], prefix + "[0].", depth + 1, max_depth)

show_structure(data)

# Specifically show the Match key which should have the stats
print("\n" + "=" * 70)
print("  MATCH DETAILS")
print("=" * 70)
match = data.get("Match", {})
if match:
    show_structure(match, "  ", max_depth=5)
