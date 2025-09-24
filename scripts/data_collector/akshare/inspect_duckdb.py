#!/usr/bin/env python3
import argparse
import duckdb
from pathlib import Path
from textwrap import indent

DEFAULT_DB = Path("/root/autodl-tmp/code/duckdb/shanghai_stock_data.duckdb")

SCHEMA_QUERY = """
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_catalog = current_database()
ORDER BY table_name, ordinal_position;
"""

TABLE_ROWCOUNT_QUERY = """
SELECT table_name, row_count
FROM (
  SELECT table_name, count(*) AS row_count
  FROM information_schema.tables t
  JOIN (
    SELECT table_name as tn FROM information_schema.tables WHERE table_type='BASE TABLE'
  ) u ON t.table_name = u.tn
  JOIN (
    SELECT 'stock_data' AS table_name UNION ALL
    SELECT 'stock_update_metadata' UNION ALL
    SELECT 'shanghai_stocks'
  ) k ON t.table_name = k.table_name
  GROUP BY table_name
) ORDER BY table_name;
"""  # simplified manual enumeration

SIMPLE_COUNTS = {
    'stock_data': 'SELECT COUNT(*) FROM stock_data',
    'stock_update_metadata': 'SELECT COUNT(*) FROM stock_update_metadata',
    'shanghai_stocks': 'SELECT COUNT(*) FROM shanghai_stocks'
}

SAMPLE_QUERIES = {
    'stock_data': """
        SELECT symbol, date, open, high, low, close, volume, change_percent
        FROM stock_data
        ORDER BY date DESC
        LIMIT {limit};
    """,
    'stock_update_metadata': """
        SELECT symbol, name, first_date, last_date, total_records, status
        FROM stock_update_metadata
        ORDER BY last_updated DESC
        LIMIT {limit};
    """,
    'shanghai_stocks': """
        SELECT symbol, name, status, retry_count, failure_reason
        FROM shanghai_stocks
        ORDER BY symbol
        LIMIT {limit};
    """
}

def format_table(rows, headers):
    if not rows:
        return '(no rows)'
    col_widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    def fmt_row(r):
        return ' | '.join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers)))
    header_line = fmt_row(headers)
    sep = '-+-'.join('-'*w for w in col_widths)
    body = '\n'.join(fmt_row(r) for r in rows)
    return f"{header_line}\n{sep}\n{body}"


def main():
    parser = argparse.ArgumentParser(description='Inspect DuckDB schema and sample data.')
    parser.add_argument('--db', type=Path, default=DEFAULT_DB, help='Path to DuckDB file')
    parser.add_argument('--limit', type=int, default=5, help='Rows to sample from each table')
    parser.add_argument('--symbols', type=str, default=None, help='Comma separated symbols to sample specifically from stock_data')
    args = parser.parse_args()

    if not args.db.exists():
        print(f"[ERROR] Database not found: {args.db}")
        return

    conn = duckdb.connect(str(args.db))
    print(f"== DuckDB File: {args.db}")

    # Discover available tables once to tailor later queries
    available_tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}

    # Show simple counts
    print("\n== Table Row Counts ==")
    for name, q in SIMPLE_COUNTS.items():
        if name not in available_tables:
            print(f"{name}: (table not found)")
            continue
        try:
            count = conn.execute(q).fetchone()[0]
            print(f"{name}: {count:,}")
        except Exception as e:
            print(f"{name}: (error: {e})")

    # Show schema per table
    print("\n== Schema Details ==")
    schema_rows = conn.execute(SCHEMA_QUERY).fetchall()
    by_table = {}
    for t, c, dt in schema_rows:
        by_table.setdefault(t, []).append((c, dt))
    for t in sorted(by_table.keys()):
        print(f"Table: {t}")
        for c, dt in by_table[t]:
            print(f"  - {c}: {dt}")

    # Sample generic queries
    for tbl, tmpl in SAMPLE_QUERIES.items():
        print(f"\n== Sample Rows: {tbl} ==")
        if tbl not in available_tables:
            print('(table not found, skipping)')
            continue
        try:
            rows = conn.execute(tmpl.format(limit=args.limit)).fetchall()
            if rows:
                headers = [d[0] for d in conn.description]
                print(format_table(rows, headers))
            else:
                print('(no rows)')
        except Exception as e:
            print(f"(error executing sample query: {e})")

    # Optional specific symbols sample
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
        placeholders = ','.join(['?']*len(symbols))
        print(f"\n== stock_data for symbols: {', '.join(symbols)} ==")
        try:
            query = f"""
                SELECT symbol, date, open, high, low, close, volume, change_percent
                FROM stock_data
                WHERE symbol IN ({placeholders})
                ORDER BY date DESC
                LIMIT {args.limit * len(symbols)}
            """
            rows = conn.execute(query, symbols).fetchall()
            if rows:
                headers = [d[0] for d in conn.description]
                print(format_table(rows, headers))
            else:
                print('(no rows for given symbols)')
        except Exception as e:
            print(f"(error: {e})")

    conn.close()

if __name__ == '__main__':
    main()
