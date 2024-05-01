CREATE OR REPLACE FUNCTION check_delivery_date()
    RETURNS TRIGGER AS
$$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM delivery_list dl
                 INNER JOIN delivery d ON dl.delivery_id = d.delivery_id
        WHERE dl.item_id = NEW.item_id
          AND d.delivery_date > (SELECT order_date FROM orders WHERE order_id = NEW.order_id)
    ) THEN
        RAISE EXCEPTION 'The item has not been delivered before the order date';
    END IF;
    RETURN NEW;
END;
$$
LANGUAGE plpgsql;

CREATE TRIGGER delivery_date_check
    BEFORE INSERT ON order_list
    FOR EACH ROW
EXECUTE FUNCTION check_delivery_date();
