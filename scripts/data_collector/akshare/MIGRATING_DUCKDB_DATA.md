# DuckDB Data Migration Guide

Comprehensive instructions for relocating the existing AkShare / Shanghai stock DuckDB database (`shanghai_stock_data.duckdb`) to a new location while keeping all current tools (`collector_standalone.py`, `duckdb_extractor.py`, `qlib_workflow_runner.py`) working without code breakage.

---
## 1. What Lives in the DuckDB File
The DuckDB file (default path: `scripts/data_collector/akshare/source/shanghai_stock_data.duckdb`) stores:

| Category | Typical Tables (stock mode) | Purpose |
|----------|----------------------------|---------|
| Price & Factors | `stock_data` | OHLCV + derived columns (amplitude, turnover, etc.) |
| Metadata | `stock_update_metadata` | Last update date, status (active, failed, etc.) |
| Symbol universe / cache | `shanghai_stocks` (and potentially other exchange tables) | Cached symbol lists & activity flags |
| Aux tables (may appear) | `jobs`, `failed_stocks`, temp staging tables | Retry & diagnostic bookkeeping |

Backup of this single file preserves the entire local history and cache state.

---
## 2. Where the Path Is Currently Hard-Coded

| File | Default Reference |
|------|-------------------|
| `scripts/data_collector/akshare/duckdb_extractor.py` | Constructor default argument `db_path="scripts/data_collector/akshare/source/shanghai_stock_data.duckdb"` |
| `scripts/data_collector/akshare/qlib_workflow_runner.py` | `self.duckdb_path = Path("scripts/data_collector/akshare/source/shanghai_stock_data.duckdb")` |
| `scripts/data_collector/akshare/collector_standalone.py` | Inside `_init_stock_data_db()` (not fully shown in snippet) – usually constructs same relative path |

If you do nothing else and only move the file, these components will fail with `FileNotFoundError` unless you also move/rename accordingly or provide a compatibility layer (environment variable or symlink).

---
## 3. Migration Strategy Options (Choose One)

| Strategy | Effort | Code Changes | Portability | Recommended When |
|----------|--------|--------------|-------------|------------------|
| A. Preserve relative path (copy tree) | Low | None | Medium | You can keep same repo layout on new host |
| B. Environment variable override | Low/Med | Tiny snippet (suggested) | High | Multiple environments (dev/prod) |
| C. Command-line / config param | Medium | Add args / config | High | You already manage YAML/env config |
| D. Symlink at old path -> new location | Low | None | Medium | You control filesystem; minimal edits |
| E. Docker volume & ENV combo | Medium | Minor | High | Containerized deployment |

You can layer B + D for maximum safety (ENV first, fallback to symlink, then default path).

---
## 4. Step-by-Step: Simple Relocation (Strategy A)
Example: Move DB to `/data/market/duckdb/shanghai_stock_data.duckdb`.

1. Ensure no process is writing: stop collectors / training jobs.
2. Create target directory:
   ```bash
   mkdir -p /data/market/duckdb
   ```
3. Copy with preservation:
   ```bash
   cp -av scripts/data_collector/akshare/source/shanghai_stock_data.duckdb /data/market/duckdb/
   ```
4. (Optional) Backup original:
   ```bash
   cp scripts/data_collector/akshare/source/shanghai_stock_data.duckdb ./shanghai_stock_data.duckdb.bak.$(date +%F)
   ```
5. Update code (choose Strategy B, C or D below) to point to new location.
6. Verify integrity (see Section 9).
7. Remove or archive old file only after successful validation.

---
## 5. Environment Variable Override (Strategy B – Recommended)

Add an environment variable `AKSHARE_DUCKDB_PATH` pointing to the new file:
```bash
export AKSHARE_DUCKDB_PATH=/data/market/duckdb/shanghai_stock_data.duckdb
```

Then (suggested patch) modify each file where the path is set. Illustrative Python snippet:
```python
import os
from pathlib import Path

DEFAULT_DUCKDB = Path("scripts/data_collector/akshare/source/shanghai_stock_data.duckdb")
DUCKDB_PATH = Path(os.environ.get("AKSHARE_DUCKDB_PATH", DEFAULT_DUCKDB))
```

In `duckdb_extractor.py` constructor:
```python
def __init__(..., db_path: str = None, ...):
    from pathlib import Path
    import os
    if db_path is None:
        db_path = os.environ.get("AKSHARE_DUCKDB_PATH", "scripts/data_collector/akshare/source/shanghai_stock_data.duckdb")
    self.db_path = Path(db_path)
```

In `qlib_workflow_runner.py` replace the direct `Path("...")` with the environment-aware version. The same pattern can be used in `_init_stock_data_db()` inside `collector_standalone.py`.

Benefit: No need to keep different branches for different servers.

---
## 6. CLI / Config Parameter (Strategy C)

If you already invoke scripts manually, add an argument:
```bash
python qlib_workflow_runner.py --duckdb-path /data/market/duckdb/shanghai_stock_data.duckdb
```

Add to `argparse` in `qlib_workflow_runner.py`:
```python
parser.add_argument("--duckdb-path", type=str, default=None, help="Override DuckDB file path")
...
runner = QlibWorkflowRunner(config_path=args.config, model_type=args.model, duckdb_path=args.duckdb_path)
```
Adjust `__init__` to accept `duckdb_path` and fallback to ENV → default path chain.

You can also extend an existing YAML config (e.g., workflow config) with a field like:
```yaml
data:
  duckdb_path: /data/market/duckdb/shanghai_stock_data.duckdb
```
Load it before constructing the runner.

---
## 7. Symlink (Strategy D)

Leave code untouched; create a symbolic link so the original relative path still resolves:
```bash
mkdir -p scripts/data_collector/akshare/source
ln -sf /data/market/duckdb/shanghai_stock_data.duckdb scripts/data_collector/akshare/source/shanghai_stock_data.duckdb
```
Pros: Zero code changes. Cons: Less explicit; may confuse future maintainers.

---
## 8. Docker / Container Deployment (Strategy E)

1. Place DB on host: `/srv/marketdata/shanghai_stock_data.duckdb`.
2. Run container with volume:
   ```bash
   docker run -d \
     -e AKSHARE_DUCKDB_PATH=/data/db/shanghai_stock_data.duckdb \
     -v /srv/marketdata:/data/db \
     yourimage:latest
   ```
3. Inside container, code automatically uses the env variable if Strategy B patch applied.
4. For read-only inference:
   ```bash
   -v /srv/marketdata:/data/db:ro
   ```

---
## 9. Verification & Sanity Checks

After migration (pick a few):

### 9.1 Basic Table Presence
```bash
python - <<'PY'
import duckdb, os
db = os.environ.get('AKSHARE_DUCKDB_PATH', 'scripts/data_collector/akshare/source/shanghai_stock_data.duckdb')
con = duckdb.connect(db)
print('Tables:', con.execute('SHOW TABLES').fetchall())
print('Stock rows:', con.execute('SELECT COUNT(*) FROM stock_data').fetchone())
print('Active symbols:', con.execute("SELECT COUNT(DISTINCT symbol) FROM stock_update_metadata WHERE status='active'").fetchone())
PY
```

### 9.2 Use `duckdb_extractor.py`
```bash
python - <<'PY'
from scripts.data_collector.akshare.duckdb_extractor import DuckDBDataExtractor
ex = DuckDBDataExtractor()  # if ENV set
info = ex.get_database_info()
print(info['total_symbols'], info['earliest_date'], info['latest_date'])
PY
```

### 9.3 Run a small workflow preparation
```bash
python scripts/data_collector/akshare/qlib_workflow_runner.py --prepare-only --start-date 2024-01-01 --end-date 2024-03-31 --symbols 600000 600036
```

Confirm that binary qlib data is regenerated in `scripts/data_collector/akshare/qlib_data`.

---
## 10. Rollback Plan

If something fails:
1. Restore backup file to original location.
2. Unset `AKSHARE_DUCKDB_PATH` (or remove overrides).
3. Re-run minimal verification (Section 9.1).
4. Investigate file permissions or partial copy issues (compare sizes with `ls -lh`).

---
## 11. Handling Concurrency & Locking

DuckDB uses single-writer, multi-reader semantics. Tips:
* Avoid running the collector (writer) simultaneously with a long training job if possible.
* For continuous ingestion, implement a staging file then atomic move (`mv staging.duckdb shanghai_stock_data.duckdb`).
* For read-only analysis nodes, distribute a snapshot copy (rsync nightly) instead of direct shared write access.

---
## 12. Backups & Versioning

Suggested cron (daily):
```bash
cp /data/market/duckdb/shanghai_stock_data.duckdb /data/market/backup/shanghai_stock_data.duckdb.$(date +%F)
find /data/market/backup -type f -mtime +14 -name 'shanghai_stock_data.duckdb.*' -delete
```
For integrity, optionally compute checksum:
```bash
sha256sum /data/market/duckdb/shanghai_stock_data.duckdb > /data/market/duckdb/duckdb.sha256
```

---
## 13. Multi-Environment Pattern

| Env | Example Path | ENV Var |
|-----|--------------|---------|
| Dev | `~/devdata/shanghai_stock_data.duckdb` | `AKSHARE_DUCKDB_PATH` |
| Staging | `/staging/market/duckdb/shanghai_stock_data.duckdb` | `AKSHARE_DUCKDB_PATH` |
| Prod | `/data/market/duckdb/shanghai_stock_data.duckdb` | `AKSHARE_DUCKDB_PATH` |

Use shell profile fragments (`/etc/profile.d/akshare.sh`) per host to set correct variable automatically.

---
## 14. Performance Considerations After Migration

| Aspect | Tip |
|--------|-----|
| I/O Throughput | Store DB on SSD/NVMe for faster extraction when training |
| Compression | DuckDB uses columnar storage; no manual compression needed |
| Large Queries | Limit symbol subset or date range during experiments |
| Memory | Use `PRAGMA threads=N;` if you parallelize heavy analytical queries |

Example (ad hoc):
```python
import duckdb, os
con = duckdb.connect(os.environ['AKSHARE_DUCKDB_PATH'])
con.execute('PRAGMA threads=4;')
```

---
## 15. Optional: Graceful Code Patch Template

Create a small helper `scripts/data_collector/akshare/db_path.py`:
```python
from pathlib import Path
import os

DEFAULT_REL = Path('scripts/data_collector/akshare/source/shanghai_stock_data.duckdb')

def get_duckdb_path():
    env = os.environ.get('AKSHARE_DUCKDB_PATH')
    if env:
        return Path(env)
    return DEFAULT_REL
```
Then in all modules:
```python
from .db_path import get_duckdb_path
self.duckdb_path = get_duckdb_path()
```

---
## 16. Common Pitfalls

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError` | ENV var path typo or missing file | `echo $AKSHARE_DUCKDB_PATH`, verify exists |
| Empty extraction result | Wrong date filter or inactive symbols only | Check `stock_update_metadata` status values |
| Workflow runner exits early | Missing data after migration | Ensure copy completed (compare file sizes) |
| Slow extraction | DB placed on network filesystem | Move to local SSD or enable caching layer |
| Permission denied | File owned by root; running as user | `chown` or adjust ACLs |

---
## 17. Quick Validation Checklist (TL;DR)
1. Set `AKSHARE_DUCKDB_PATH` to new absolute path.
2. Run Section 9.1 script – tables list OK? counts reasonable?
3. Run extractor small query (Section 9.2).
4. Run `--prepare-only` workflow (Section 9.3).
5. Start a training run; confirm no errors.
6. Remove old DB or keep as dated backup.

---
## 18. Summary
Relocating the DuckDB file is safe and simple if you add a flexible indirection layer (environment variable or helper function). Prefer **Strategy B (ENV)** for long-term maintainability; pair with automated backups and a validation script executed in CI or pre-training hooks.

---
Feel free to extend this guide with organization-specific deployment notes.
