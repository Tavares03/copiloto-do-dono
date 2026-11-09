import pytest

from copiloto.agent.tools.sql_guard import UnsafeQueryError, validate_sql


def rejected(sql: str) -> str:
    with pytest.raises(UnsafeQueryError) as info:
        validate_sql(sql, max_rows=200)
    return str(info.value)


# --- escrita e DDL --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "UPDATE orders SET total = 0",
        "INSERT INTO orders (order_id) VALUES (1)",
        "DROP TABLE orders",
        "DROP VIEW analytics.orders",
        "TRUNCATE orders",
        "CREATE TABLE hack (id int)",
        "ALTER TABLE orders ADD COLUMN x int",
        "GRANT ALL ON orders TO public",
        "COPY orders TO '/tmp/vazamento.csv'",
        "SET statement_timeout = 0",
        "EXPLAIN ANALYZE SELECT * FROM orders",
        "MERGE INTO orders USING menu ON true WHEN MATCHED THEN DELETE",
    ],
)
def test_rejects_anything_that_is_not_a_select(sql: str):
    assert "SELECT" in rejected(sql)


def test_rejects_delete_hidden_inside_a_cte():
    assert "SELECT" in rejected("WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d")


def test_rejects_select_into_and_row_locks():
    rejected("SELECT * INTO hack FROM orders")
    rejected("SELECT * FROM orders FOR UPDATE")


# --- múltiplas consultas ------------------------------------------------------------------


def test_rejects_multiple_statements():
    assert "uma consulta" in rejected("SELECT 1; DROP TABLE orders")
    assert "uma consulta" in rejected("SELECT 1; SELECT 2")


def test_comment_with_semicolon_is_a_single_harmless_statement():
    result = validate_sql("SELECT 1 -- ; DROP TABLE orders", max_rows=200)

    assert result.sql == "SELECT 1 LIMIT 200"  # comentário removido antes de ir ao banco


def test_rejects_empty_and_invalid_sql():
    rejected("")
    rejected("   ;  ")
    rejected("SELEC * FORM orders")


# --- whitelist de tabelas ------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM core.orders",
        "SELECT * FROM public.orders",
        "SELECT * FROM pg_catalog.pg_shadow",
        "SELECT usename, passwd FROM pg_shadow",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_user",
        "SELECT * FROM orders o JOIN core.order_items i ON i.order_id = o.order_id",
        "SELECT * FROM (SELECT * FROM core.orders) AS sub",
        "SELECT * FROM outro_banco.analytics.orders",
    ],
)
def test_rejects_tables_outside_the_whitelist(sql: str):
    assert "não permitida" in rejected(sql)


# --- funções perigosas ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_sleep(10)",
        "SELECT set_config('statement_timeout', '0', false)",
        "SELECT current_setting('data_directory')",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM dblink('host=evil', 'select 1') AS t(a int)",
        "SELECT lo_import('/etc/passwd')",
        "SELECT query_to_xml('select * from core.orders', true, true, '')",
        "SELECT nextval('core.orders_id_seq')",
    ],
)
def test_rejects_dangerous_functions(sql: str):
    assert "não permitida" in rejected(sql)


# --- consultas válidas ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT sum(net_revenue) FROM orders WHERE order_date = DATE '2026-10-31'",
        "SELECT * FROM analytics.orders",
        'SELECT * FROM "orders"',
        "SELECT o.channel, count(*) FROM orders o JOIN sold_items s USING (order_id) GROUP BY 1",
        "WITH semana AS (SELECT * FROM orders WHERE order_date > DATE '2026-10-24') "
        "SELECT count(*) FROM semana",
        "SELECT item_name FROM menu UNION SELECT channel FROM channels",
        "SELECT d::date FROM generate_series(DATE '2026-10-01', DATE '2026-10-07', "
        "INTERVAL '1 day') AS d",
        "SELECT day, name FROM holidays",
        "SELECT * FROM orders WHERE order_id IN (SELECT order_id FROM sold_items)",
    ],
)
def test_accepts_read_only_queries_on_whitelisted_tables(sql: str):
    result = validate_sql(sql, max_rows=200)

    assert result.sql.upper().startswith(("SELECT", "WITH"))


def test_adds_limit_when_missing():
    assert validate_sql("SELECT * FROM orders", max_rows=200).limit == 200
    assert validate_sql("SELECT * FROM orders", max_rows=200).sql.endswith("LIMIT 200")


def test_caps_limit_that_is_too_large_or_unbounded():
    assert validate_sql("SELECT * FROM orders LIMIT 100000", max_rows=200).limit == 200
    assert validate_sql("SELECT * FROM orders LIMIT ALL", max_rows=200).limit == 200


def test_keeps_smaller_limit():
    result = validate_sql("SELECT * FROM orders ORDER BY total DESC LIMIT 5", max_rows=200)

    assert result.limit == 5
    assert result.sql.endswith("LIMIT 5")


def test_limit_applies_to_whole_union():
    result = validate_sql("SELECT item_name FROM menu UNION SELECT channel FROM channels", 50)

    assert result.sql.endswith("LIMIT 50")


def test_reports_tables_used():
    result = validate_sql(
        "SELECT * FROM orders o JOIN analytics.sold_items s USING (order_id)", max_rows=10
    )

    assert result.tables == ["orders", "sold_items"]
