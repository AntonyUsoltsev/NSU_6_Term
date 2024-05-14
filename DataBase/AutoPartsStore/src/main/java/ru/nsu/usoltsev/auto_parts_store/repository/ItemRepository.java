package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Item;

import java.sql.Timestamp;
import java.util.List;

public interface ItemRepository extends JpaRepository<Item, Long> {
    @Query("SELECT i " +
            "FROM Item i " +
            "LEFT JOIN ItemCategory ic on i.categoryId = ic.categoryId " +
            "WHERE ic.categoryName = :category")
    List<Item> findByCategory(@Param("category") String category);

    @Query("SELECT i.name, i.amount, i.cellNumber " +
            "FROM Item i ")
    List<Object[]> findAllItemsInfo();

    @Query("SELECT i.name AS item_name, SUM(ol.amount) AS total_sold " +
            "FROM Item i " +
            "JOIN OrderList ol ON i.itemId = ol.itemId " +
            "GROUP BY i.name " +
            "ORDER BY total_sold DESC " +
            "LIMIT 10")
    List<Object[]> getTopTenSoldDetails();


    @Query("SELECT s.name, i.price, d.deliveryDate " +
            "FROM Item i " +
            "LEFT JOIN DeliveryList dl on i.itemId = dl.itemId " +
            "LEFT JOIN Delivery d on dl.deliveryId = d.deliveryId " +
            "LEFT JOIN Supplier s on d.supplierId = s.supplierId " +
            "WHERE i.name = :item ")
    List<Object[]> getSupplierPriceDateForItem(@Param("item") String item);

    @Query("SELECT i.name " +
            "FROM Item i ")
    List<String> getItemsNames();


    @Query("SELECT i.name, i.defectAmount, d.deliveryDate, s.name " +
            "FROM Item i " +
            "JOIN DeliveryList dl ON i.itemId = dl.itemId " +
            "JOIN Delivery d ON dl.deliveryId = d.deliveryId " +
            "JOIN Supplier s ON d.supplierId = s.supplierId " +
            "WHERE i.defectAmount > 0 AND " +
            ":fromDate <= d.deliveryDate AND d.deliveryDate <= :toDate ")
    List<Object[]> findDefectItems(@Param("fromDate") Timestamp fromDate,
                                   @Param("toDate") Timestamp toDate);

    @Query("SELECT i.name, ic.categoryName " +
            "FROM Item i " +
            "LEFT JOIN ItemCategory ic on i.categoryId = ic.categoryId")
    List<Object[]> getItemsCatalog();

    @Query("SELECT i.amount, i.price, s.name " +
            "FROM Item i " +
            "LEFT JOIN DeliveryList dl on i.itemId = dl.itemId " +
            "LEFT JOIN Delivery d on dl.deliveryId = d.deliveryId " +
            "LEFT JOIN Supplier s on d.supplierId = s.supplierId " +
            "WHERE i.name = :name ")
    List<Object[]> getSupplierItemInfo(@Param("name")  String name);


    @Query("SELECT count (distinct i.cellNumber) " +
            "FROM Item i " )
    Integer findStoreCapacity();
}
