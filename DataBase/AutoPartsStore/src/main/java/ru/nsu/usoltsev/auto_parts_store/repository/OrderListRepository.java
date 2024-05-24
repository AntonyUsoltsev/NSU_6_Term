package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.OrderList;

import java.util.List;

public interface OrderListRepository extends JpaRepository<OrderList, OrderList.OrderListKey> {
    @Query("SELECT ol " +
            "FROM OrderList  ol " +
            "WHERE ol.orderId = :orderId")
    List<OrderList> findOrdersByOrderId(@Param("orderId") Long orderId);

    @Query("SELECT SUM(ol.amount * i.price) " +
            "FROM OrderList  ol " +
            "JOIN Item i ON ol.itemId = i.itemId " +
            "WHERE ol.orderId = :orderId")
    Integer getOrderFullPrice(@Param("orderId") Long orderId);
}

