"""
Build DuckDB database for CreditScope Pro.

This script:
1. Creates the DuckDB database file
2. Reads schema.sql
3. Executes all CREATE TABLE statements
4. Confirms the tables were created
"""

from pathlib import Path
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "creditscope.duckdb"
SCHEMA_PATH = PROJECT_ROOT / "src" / "database" / "schema.sql"


def build_database() -> None:
    """Create DuckDB database and apply schema."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    schema_sql = SCHEMA_PATH.read_text()

    with duckdb.connect(str(DB_PATH)) as conn:
        conn.execute(schema_sql)

        tables = conn.execute("SHOW TABLES").fetchall()

    print(f"Database created at: {DB_PATH}")
    print("Tables created:")

    for table in tables:
        print(f"- {table[0]}")


if __name__ == "__main__":
    build_database()
