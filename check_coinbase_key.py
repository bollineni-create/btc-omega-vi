#!/usr/bin/env python3
"""
Check Coinbase API key format for Advanced Trade (fix 401 Unauthorized).
Run from project dir: python check_coinbase_key.py

Loads .env and/or cdp_api_key.json, then reports whether the key is in the
format the API expects (full key name: organizations/{org_id}/apiKeys/{key_id}).
"""
import json
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

def main():
    key = os.environ.get("COINBASE_API_KEY")
    cdp_path = os.environ.get("COINBASE_CDP_KEY_JSON", "cdp_api_key.json")
    if not key and os.path.isfile(cdp_path):
        try:
            with open(cdp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = (data.get("name") or data.get("id") or "").strip()
        except Exception as e:
            print(f"Cannot read key from {cdp_path}: {e}", file=sys.stderr)
            sys.exit(1)

    if not key:
        print("No Coinbase API key found. Set COINBASE_API_KEY in .env or use cdp_api_key.json with 'id' or 'name'.")
        sys.exit(1)

    org_id = os.environ.get("COINBASE_ORG_ID", "").strip()
    full_name = key

    if not key.startswith("organizations/"):
        if org_id:
            full_name = f"organizations/{org_id}/apiKeys/{key}"
            print("Key format: OK (full key name built from COINBASE_ORG_ID + key id)")
            print("  Using:", full_name[:50] + "..." if len(full_name) > 50 else full_name)
        else:
            print("Key format: INCOMPLETE (API expects full key name, not just key id)")
            print("  Current key (id only):", key[:20] + "..." if len(key) > 20 else key)
            print()
            print("To fix 401 Unauthorized:")
            print("  1. Open https://cloud.coinbase.com/access/api")
            print("  2. Find your organization ID (e.g. in the URL or in the key details).")
            print("  3. Add this line to your .env file:")
            print()
            print("     COINBASE_ORG_ID=your_organization_id_here")
            print()
            print("   Keep COINBASE_API_KEY as your key id (or use cdp_api_key.json);")
            print("   the app will build the full key name automatically.")
            sys.exit(1)
    else:
        print("Key format: OK (full key name already set)")
        print("  Using: organizations/.../apiKeys/...")

if __name__ == "__main__":
    main()
