SELECT item.item_id, item.amount, item.name, d.delivery_date, item.price
From item
         LEFT JOIN delivery_list dl on item.item_id = dl.item_id
         LEFT JOIN delivery d on dl.delivery_id = d.delivery_id

WHERE d.delivery_date<'2024-04-20 13:10:00.000000'


--     '2024-03-18 13:00:00.000000'