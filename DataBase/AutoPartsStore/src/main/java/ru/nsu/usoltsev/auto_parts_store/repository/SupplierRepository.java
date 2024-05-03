package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Supplier;

import java.util.List;

public interface SupplierRepository extends JpaRepository<Supplier, Long> {

    @Query("SELECT s " +
            "FROM Supplier s " +
            "JOIN SupplierType st on s.typeId = st.typeId " +
            "WHERE st.typeName = :type")
    List<Supplier> findSuppliersByType(@Param("type") String type);

    @Query("SELECT s " +
            "FROM Supplier s " +
            "JOIN Delivery d ON s.supplierId = d.supplierId " +
            "JOIN DeliveryList dl ON d.deliveryId = dl.deliveryId " +
            "JOIN Item i ON dl.itemId = i.itemId " +
            "JOIN ItemCategory ic ON i.categoryId = ic.categoryId " +
            "WHERE ic.categoryName = :category")
    List<Supplier> findSuppliersByItemCategory(@Param("category") String category);

    @Query("SELECT COUNT (DISTINCT s) " +
            "FROM Supplier s " +
            "JOIN SupplierType st on s.typeId = st.typeId " +
            "WHERE st.typeName = :type")
    Integer findSuppliersCountByType(@Param("type") String type);

}
