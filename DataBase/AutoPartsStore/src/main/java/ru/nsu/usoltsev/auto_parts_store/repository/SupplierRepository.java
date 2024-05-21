package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Supplier;

import java.sql.Timestamp;
import java.util.List;

public interface SupplierRepository extends JpaRepository<Supplier, Long> {


    @Query("SELECT new ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierDto(s.supplierId, s.name, s.documents, st.typeName , s.garanty) " +
            "FROM Supplier s " +
            "LEFT JOIN SupplierType st on s.typeId = st.typeId ")
    List<SupplierDto> findAllSuplliers();

    @Query("SELECT new ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierDto(s.supplierId, s.name, s.documents, 'someType' , s.garanty) " +
            "FROM Supplier s " +
            "WHERE s.typeId = :type")
    List<SupplierDto> findSuppliersByType(@Param("type") Long type);

    @Query("SELECT DISTINCT s.name, s.documents, st.typeName, s.garanty " +
            "FROM Supplier s " +
            "JOIN SupplierType st on s.typeId = st.typeId " +
            "JOIN Delivery d ON s.supplierId = d.supplierId " +
            "JOIN DeliveryList dl ON d.deliveryId = dl.deliveryId " +
            "JOIN Item i ON dl.itemId = i.itemId " +
            "JOIN ItemCategory ic ON i.categoryId = ic.categoryId " +
            "WHERE ic.categoryName = :category")
    List<Object[]> findSuppliersByItemCategory(@Param("category") String category);

    @Query("SELECT COUNT (DISTINCT s) " +
            "FROM Supplier s " +
            "WHERE s.typeId = :type")
    Integer findSuppliersCountByType(@Param("type") Long type);


    @Query("SELECT new ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierDto(s.supplierId, s.name, s.documents, st.typeName, s.garanty)  " +
            "FROM Supplier s " +
            "JOIN SupplierType st on s.typeId = st.typeId " +
            "JOIN Delivery d ON s.supplierId = d.supplierId " +
            "JOIN DeliveryList dl ON d.deliveryId = dl.deliveryId " +
            "JOIN Item i ON dl.itemId = i.itemId " +
            "WHERE i.name = :item AND " +
            "dl.amount >= :amount AND " +
            ":fromDate <= d.deliveryDate AND d.deliveryDate <= :toDate ")
    List<SupplierDto> findSuppliersByDelivery(@Param("fromDate") Timestamp fromDate,
                                              @Param("toDate") Timestamp toDate,
                                              @Param("amount") Integer amount,
                                              @Param("item") String item);
}
