-- Camada "analytics": a ÚNICA parte do banco visível para o agente (usuário read-only).
-- Para trocar a fonte de dados (ex.: ComandaHub), reescreva estas views mantendo as colunas:
-- o agente, as métricas e as evals continuam funcionando sem mudança.
--
-- Convenções:
--   * datas e horas no fuso America/Sao_Paulo;
--   * colunas de dinheiro derivadas (net_revenue, items_cost, contribution_margin) são NULL
--     em pedidos cancelados, então SUM/AVG já ignoram cancelados sem precisar de filtro;
--   * sold_items só contém itens de pedidos concluídos.

CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE VIEW analytics.channels AS
SELECT id             AS channel_id,
       code           AS channel,
       name           AS channel_name,
       commission_pct
FROM core.channels;

CREATE OR REPLACE VIEW analytics.menu AS
SELECT id                                        AS item_id,
       name                                      AS item_name,
       category,
       price,
       cost,
       price - cost                              AS unit_margin,
       round((price - cost) / price * 100, 1)    AS margin_pct,
       active,
       launched_on
FROM core.menu_items;

CREATE OR REPLACE VIEW analytics.holidays AS
SELECT day, name, kind
FROM core.holidays;

CREATE OR REPLACE VIEW analytics.orders AS
WITH costs AS (
    SELECT order_id,
           sum(quantity * unit_cost) AS items_cost,
           sum(quantity)             AS items_qty
    FROM core.order_items
    GROUP BY order_id
), base AS (
    SELECT o.*,
           c.code                                    AS channel,
           o.created_at AT TIME ZONE 'America/Sao_Paulo' AS local_ts
    FROM core.orders o
    JOIN core.channels c ON c.id = o.channel_id
)
SELECT b.id                                      AS order_id,
       b.channel,
       b.local_ts                                AS created_at_local,
       b.local_ts::date                          AS order_date,
       extract(hour FROM b.local_ts)::int        AS order_hour,
       extract(isodow FROM b.local_ts)::int      AS weekday,       -- 1 = segunda ... 7 = domingo
       (ARRAY['segunda', 'terça', 'quarta', 'quinta', 'sexta', 'sábado', 'domingo'])
           [extract(isodow FROM b.local_ts)::int] AS weekday_name,
       CASE
           WHEN extract(hour FROM b.local_ts) < 15 THEN 'almoco'
           WHEN extract(hour FROM b.local_ts) < 18 THEN 'tarde'
           ELSE 'jantar'
       END                                       AS time_slot,
       b.status,
       b.cancel_reason,
       b.subtotal,
       b.discount,
       b.delivery_fee,
       b.channel_fee,
       b.total,
       CASE WHEN b.status = 'concluido' THEN b.subtotal - b.discount END AS net_revenue,
       CASE WHEN b.status = 'concluido' THEN c.items_cost END            AS items_cost,
       CASE WHEN b.status = 'concluido'
            THEN b.subtotal - b.discount - c.items_cost - b.channel_fee
       END                                       AS contribution_margin,
       c.items_qty,
       b.prep_time_min
FROM base b
JOIN costs c ON c.order_id = b.id;

CREATE OR REPLACE VIEW analytics.sold_items AS
SELECT oi.id                                     AS line_id,
       o.order_id,
       o.order_date,
       o.order_hour,
       o.weekday,
       o.weekday_name,
       o.time_slot,
       o.channel,
       m.id                                      AS item_id,
       m.name                                    AS item_name,
       m.category,
       oi.quantity,
       oi.unit_price,
       oi.unit_cost,
       oi.quantity * oi.unit_price               AS line_revenue,  -- bruta, antes do desconto do pedido
       oi.quantity * oi.unit_cost                AS line_cost,
       oi.quantity * (oi.unit_price - oi.unit_cost) AS line_margin
FROM core.order_items oi
JOIN analytics.orders o ON o.order_id = oi.order_id
JOIN core.menu_items m  ON m.id = oi.menu_item_id
WHERE o.status = 'concluido';
