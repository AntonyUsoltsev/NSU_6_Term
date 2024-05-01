SELECT orders.order_id, i.item_id, ol.amount, i.name, orders.order_date
From orders
         LEFT JOIN order_list ol on orders.order_id = ol.order_id
         LEFT JOIN items i on ol.item_id = i.item_id
