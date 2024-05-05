package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Customer;

import java.sql.Timestamp;
import java.util.List;

public interface CustomerRepository extends JpaRepository<Customer, Long> {

    @Query("SELECT c " +
            "FROM Customer c " +
            "JOIN Orders o ON c.customerId = o.customerId " +
            "JOIN OrderList ol ON o.orderId = ol.orderId " +
            "JOIN Item i ON ol.itemId = i.itemId " +
            "WHERE i.name = :item AND " +
            "ol.amount >= :amount AND " +
            ":fromDate <= o.orderDate AND o.orderDate <= :toDate ")
    List<Customer> findCustomerByItem(@Param("fromDate") Timestamp fromDate,
                                      @Param("toDate") Timestamp toDate,
                                      @Param("amount") Integer amount,
                                      @Param("item") String item);
}


