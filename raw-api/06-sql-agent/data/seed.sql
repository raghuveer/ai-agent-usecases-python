-- Seed schema + data for UC6 sql-agent.
-- A tiny store: customers and the orders they placed.
-- The app rebuilds this DB from scratch on startup, so it stays deterministic.

CREATE TABLE customers (
    id   INTEGER PRIMARY KEY,
    name TEXT    NOT NULL,
    city TEXT    NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    amount      REAL    NOT NULL,
    created     TEXT    NOT NULL  -- ISO date, e.g. 2024-01-15
);

INSERT INTO customers (id, name, city) VALUES
    (1, 'Alice Martin',  'London'),
    (2, 'Bob Chen',      'Toronto'),
    (3, 'Carla Diaz',    'Madrid'),
    (4, 'Dan Okoro',     'Lagos'),
    (5, 'Erika Novak',   'Prague');

INSERT INTO orders (id, customer_id, amount, created) VALUES
    (1, 1, 120.50, '2024-01-15'),
    (2, 1,  75.00, '2024-02-03'),
    (3, 2, 240.00, '2024-01-22'),
    (4, 3,  18.99, '2024-03-10'),
    (5, 5, 560.25, '2024-02-28');
