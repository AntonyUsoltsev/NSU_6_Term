SELECT orders.order_id, i.item_id, ol.amount, i.name, orders.order_date
From orders
         LEFT JOIN order_list ol on orders.order_id = ol.order_id
         LEFT JOIN item i on ol.item_id = i.item_id


SELECT i.name AS item_name, SUM(ol.amount) AS total_sold
FROM item i JOIN order_list ol ON i.item_id = ol.item_id
GROUP BY i.name
ORDER BY total_sold DESC
LIMIT 10;