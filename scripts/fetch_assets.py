#!/usr/bin/env python3
################################################################################
# Qallow Asset Downloader
#
# Automatically fetches required data files, models, and dependencies that are
# too large to include in git. Caches locally to avoid re-downloading.
#
# Usage:
#   python3 scripts/fetch_assets.py [--force] [--no-cache]
#
################################################################################

from pathlib import Path
from typing import Dict, List, Optional

# Configuration
ASSETS_DIR = Path(__file__).parent.parent / "data" / "assets"
CACHE_DIR = Path.home() / ".cache" / "qallow"
CONFIG_FILE = Path(__file__).parent.parent / "scripts" / "assets.json"

COLORS = {
    "BLUE": "\033[0;34m",
    "GREEN": "\033[0;32m",
    "YELLOW": "\033[1;33m",
    "RED": "\033[0;31m",
    "NC": "\033[0m"
}

def colored(text: str, color: str) -> str:
    """Return colored text for terminal output."""
    return f"{COLORS.get(color, '')}{text}{COLORS['NC']}"

def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def compute_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    hash_obj = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def download_file(url: str, dest: Path, expected_hash: Optional[str] = None) -> bool:
    """Download a file with progress reporting."""
    try:
        print(f"  Downloading {url}...")
        
        # Simple progress callback
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, int(100.0 * downloaded / total_size)) if total_size > 0 else 0
            print(f"\r    [{percent:3d}%] {downloaded:,} / {total_size:,} bytes", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest, reporthook=report_progress)
        print()  # New line after progress
        
        # Verify hash if provided
        if expected_hash:
            actual_hash = compute_hash(dest)
            if actual_hash != expected_hash:
                print(colored(f"  ✗ Hash mismatch for {dest.name}", "RED"))
                print(f"    Expected: {expected_hash}")
                print(f"    Got:      {actual_hash}")
                return False
        
        return True
    except Exception as e:
        print(colored(f"  ✗ Download failed: {e}", "RED"))
        return False

def load_assets_config() -> Dict:
    """Load assets configuration from JSON."""
    if not CONFIG_FILE.exists():
        print(colored(f"⚠ Config not found: {CONFIG_FILE}", "YELLOW"))
        return {"assets": []}
    
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def fetch_asset(asset: Dict, force: bool = False, no_cache: bool = False) -> bool:
    """Fetch a single asset."""
    name = asset.get("name", "unknown")
    url = asset.get("url")
    dest_path = asset.get("dest", name)
    hash_value = asset.get("hash")
    optional = asset.get("optional", False)
    
    if not url:
        print(colored(f"  ✗ No URL for asset: {name}", "RED"))
        return not optional
    
    # Resolve paths
    abs_dest = ASSETS_DIR / dest_path
    ensure_dir(abs_dest.parent)
    
    # Check if file already exists
    if abs_dest.exists() and not force:
        if hash_value:
            actual_hash = compute_hash(abs_dest)
            if actual_hash == hash_value:
                print(colored(f"  ✓ {name} (cached)", "GREEN"))
                return True
        else:
            print(colored(f"  ✓ {name} (already exists)", "GREEN"))
            return True
    
    # Try cache first if not --no-cache
    if not no_cache and hash_value:
        cache_path = CACHE_DIR / hash_value
        if cache_path.exists():
            print(f"  Using cache for {name}...")
            shutil.copy2(cache_path, abs_dest)
            print(colored(f"  ✓ {name} (from cache)", "GREEN"))
            return True
    
    # Download
    if download_file(url, abs_dest, hash_value):
        # Cache it
        if hash_value and not no_cache:
            ensure_dir(CACHE_DIR)
            cache_path = CACHE_DIR / hash_value
            if not cache_path.exists():
                shutil.copy2(abs_dest, cache_path)
        
        print(colored(f"  ✓ {name}", "GREEN"))
        return True
    else:
        if optional:
            print(colored(f"  ⚠ Optional asset failed: {name}", "YELLOW"))
            return True
        else:
            print(colored(f"  ✗ Required asset failed: {name}", "RED"))
            return False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qallow Asset Downloader")
    parser.add_argument("--force", action="store_true", help="Force re-download of all assets")
    parser.add_argument("--no-cache", action="store_true", help="Don't use or create cache")
    parser.add_argument("--list", action="store_true", help="List all assets")
    
    args = parser.parse_args()
    
    print(colored("═" * 80, "BLUE"))
    print(colored("Qallow Asset Downloader", "BLUE"))
    print(colored("═" * 80, "BLUE"))
    print()
    
    # Create directories
    ensure_dir(ASSETS_DIR)
    
    # Load configuration
    config = load_assets_config()
    assets = config.get("assets", [])
    
    if args.list:
        print("Available assets:")
        for asset in assets:
            optional_tag = " (optional)" if asset.get("optional", False) else ""
            print(f"  - {asset.get('name', 'unknown')}{optional_tag}")
            print(f"    URL: {asset.get('url', 'N/A')}")
            print(f"    Dest: {asset.get('dest', 'N/A')}")
            if asset.get('hash'):
                print(f"    Hash: {asset.get('hash')[:16]}...")
        print()
        return 0
    
    if not assets:
        print(colored("No assets to download (config empty or missing)", "YELLOW"))
        return 0
    
    print(f"Fetching {len(assets)} asset(s)...")
    print()
    
    failed = []
    for i, asset in enumerate(assets, 1):
        name = asset.get("name", f"asset_{i}")
        print(f"[{i}/{len(assets)}] {name}")
        
        if not fetch_asset(asset, force=args.force, no_cache=args.no_cache):
            failed.append(name)
    
    print()
    print(colored("═" * 80, "BLUE"))
    
    if failed:
        print(colored(f"⚠ {len(failed)} asset(s) failed:", "YELLOW"))
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print(colored("✓ All assets ready!", "GREEN"))
        print()
        print(f"Assets directory: {ASSETS_DIR}")
        print(f"Cache directory:  {CACHE_DIR}")
        return 0

if __name__ == "__main__":
    sys.exit(main())
