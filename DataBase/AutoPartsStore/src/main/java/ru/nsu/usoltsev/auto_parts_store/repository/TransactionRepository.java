package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Transaction;

import java.sql.Timestamp;
import java.util.List;

public interface TransactionRepository extends JpaRepository<Transaction, Long> {

    @Query("SElECT DISTINCT i.name, SUM(ol.amount), SUM(ol.amount*i.price) " +
            "FROM Transaction t " +
            "LEFT JOIN Orders o on t.orderId = o.orderId " +
            "LEFT JOIN OrderList ol on o.orderId = ol.orderId " +
            "LEFT JOIN Item i on ol.itemId = i.itemId " +
            "WHERE to_char(o.orderDate,'yyyy/MM/dd') = :day "+
            "GROUP BY i.name" )
    public List<Object[]> findRealiseItemsByDay(@Param("day") String day);

}
