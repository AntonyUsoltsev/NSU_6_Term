package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.CashReportDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Transaction;

import java.sql.Timestamp;
import java.util.List;

public interface TransactionRepository extends JpaRepository<Transaction, Long> {

    @Query("SElECT DISTINCT i.name, SUM(ol.amount), SUM(ol.amount*i.price) " +
            "FROM Transaction t " +
            "LEFT JOIN Orders o on t.orderId = o.orderId " +
            "LEFT JOIN OrderList ol on o.orderId = ol.orderId " +
            "LEFT JOIN Item i on ol.itemId = i.itemId " +
            "WHERE to_char(o.orderDate,'yyyy/MM/dd') = :day " +
            "GROUP BY i.name")
    List<Object[]> findRealiseItemsByDay(@Param("day") String day);


    @Query("SELECT i.name , t.transactionDate, d.deliveryDate, t.transactionDate - d.deliveryDate " +
            "FROM Transaction t " +
            "LEFT JOIN Orders o on t.orderId = o.orderId " +
            "LEFT JOIN OrderList ol on o.orderId = ol.orderId " +
            "LEFT JOIN Item i on ol.itemId = i.itemId " +
            "LEFT JOIN DeliveryList dl on i.itemId = dl.itemId " +
            "LEFT JOIN Delivery d on dl.deliveryId = d.deliveryId ")
    List<Object[]> findSellingSpeed();

    @Query("SELECT i.name, ol.amount, s.name, i.price * ol.amount " +
            "FROM Transaction t " +
            "LEFT JOIN Orders o on t.orderId = o.orderId " +
            "LEFT JOIN OrderList ol on o.orderId = ol.orderId " +
            "LEFT JOIN Item i on ol.itemId = i.itemId " +
            "LEFT JOIN DeliveryList dl on i.itemId = dl.itemId " +
            "LEFT JOIN Delivery d on dl.deliveryId = d.deliveryId " +
            "LEFT JOIN Supplier s on d.supplierId = s.supplierId " +
            "WHERE t.transactionId = :transactionId")
    List<Object[]> findTransactionOrderList(@Param("transactionId") Long transactionId);

    @Query("SELECT t.transactionId, t.transactionDate, tt.typeName, o.fullPrice, c.cashierId, c.name, c.secondName, cus.name, cus.secondName, cus.email " +
            "FROM Transaction t " +
            "LEFT JOIN Cashier c on t.cashierId = c.cashierId " +
            "LEFT JOIN TransactionType tt on  t.typeId = tt.typeId " +
            "LEFT JOIN Orders o on t.orderId = o.orderId " +
            "LEFT JOIN Customer cus on o.customerId = cus.customerId " +
            "WHERE :fromDate <= t.transactionDate AND t.transactionDate <= :toDate " )
    List<Object[]> findTransactionInfo(@Param("fromDate") Timestamp fromDate,
                                       @Param("toDate")Timestamp toDate);

    @Query("SELECT ic.categoryName, sum(ol.amount) " +
            "FROM Transaction t " +
            "LEFT JOIN Orders  o on t.orderId = o.orderId " +
            "LEFT JOIN OrderList ol on o.orderId = ol.orderId " +
            "LEFT JOIN Item i on ol.itemId = i.itemId " +
            "LEFT JOIN ItemCategory ic on i.categoryId = ic.categoryId " +
            "GROUP BY ic.categoryName")
    List<Object[]> findAverageSell();
}
