INSERT INTO orders (order_id, full_price, order_date, customer_id)
VALUES (1, 0, '2024-03-18 13:00:00.000000', 1);

UPDATE orders
SET full_price = (SELECT SUM(ol.amount * i.price)
                  FROM order_list ol
                           JOIN items i ON ol.item_id = i.item_id
                  WHERE ol.order_id = orders.order_id)
WHERE order_id = 1;

INSERT INTO orders (order_id, full_price, order_date, customer_id)
VALUES (2, 0, '2024-01-10 13:00:00.000000', 2);

UPDATE orders
SET full_price = (SELECT SUM(ol.amount * i.price)
                  FROM order_list ol
                           JOIN items i ON ol.item_id = i.item_id
                  WHERE ol.order_id = orders.order_id)
WHERE order_id = 2;

INSERT INTO orders (order_id, full_price, order_date, customer_id)
VALUES (3, 0, '2024-04-20 13:10:00.000000', 3);

UPDATE orders
SET full_price = (SELECT SUM(ol.amount * i.price)
                  FROM order_list ol
                           JOIN items i ON ol.item_id = i.item_id
                  WHERE ol.order_id = orders.order_id)
WHERE order_id = 3;