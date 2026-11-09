-- Camada "core": dados brutos do restaurante.
-- Hoje vem do gerador sintético; no futuro, do ComandaHub.
-- O agente NUNCA lê daqui: ele só enxerga as views do schema analytics.

CREATE SCHEMA IF NOT EXISTS core;

CREATE TABLE IF NOT EXISTS core.channels (
    id             smallint PRIMARY KEY,
    code           text UNIQUE NOT NULL,          -- ifood | 99food | balcao | whatsapp
    name           text NOT NULL,
    commission_pct numeric(5, 2) NOT NULL         -- comissão sobre o subtotal (valores ilustrativos)
);

CREATE TABLE IF NOT EXISTS core.menu_items (
    id          serial PRIMARY KEY,
    name        text UNIQUE NOT NULL,
    category    text NOT NULL
                CHECK (category IN ('burger', 'acompanhamento', 'bebida', 'sobremesa')),
    price       numeric(10, 2) NOT NULL,          -- preço atual
    cost        numeric(10, 2) NOT NULL,          -- custo (CMV) atual
    active      boolean NOT NULL DEFAULT true,
    launched_on date NOT NULL
);

CREATE TABLE IF NOT EXISTS core.orders (
    id            bigserial PRIMARY KEY,
    channel_id    smallint NOT NULL REFERENCES core.channels (id),
    created_at    timestamptz NOT NULL,
    status        text NOT NULL CHECK (status IN ('concluido', 'cancelado')),
    cancel_reason text,
    subtotal      numeric(10, 2) NOT NULL,
    discount      numeric(10, 2) NOT NULL DEFAULT 0,
    delivery_fee  numeric(10, 2) NOT NULL DEFAULT 0, -- repassada ao entregador, não é receita
    channel_fee   numeric(10, 2) NOT NULL DEFAULT 0, -- comissão cobrada pelo canal (valor da época)
    total         numeric(10, 2) NOT NULL,           -- o que o cliente pagou
    prep_time_min smallint,                          -- NULL se cancelado antes do preparo
    CHECK ((status = 'cancelado') = (cancel_reason IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS orders_created_at_idx ON core.orders (created_at);

CREATE TABLE IF NOT EXISTS core.order_items (
    id           bigserial PRIMARY KEY,
    order_id     bigint NOT NULL REFERENCES core.orders (id) ON DELETE CASCADE,
    menu_item_id int NOT NULL REFERENCES core.menu_items (id),
    quantity     smallint NOT NULL CHECK (quantity > 0),
    unit_price   numeric(10, 2) NOT NULL,          -- preço no momento da venda
    unit_cost    numeric(10, 2) NOT NULL           -- custo no momento da venda
);

CREATE INDEX IF NOT EXISTS order_items_order_id_idx ON core.order_items (order_id);

CREATE TABLE IF NOT EXISTS core.holidays (
    day  date PRIMARY KEY,
    name text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('feriado', 'data_comemorativa'))
);
