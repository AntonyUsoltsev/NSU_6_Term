package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Supplier;

import java.util.List;

public interface SupplierRepository extends JpaRepository<Supplier, Long> {
    @Query(value =
            "SELECT s, i.category " +
                    "FROM Supplier s " +
                    "JOIN Delivery d ON s.supplier_id = d.supplier_id " +
                    "JOIN delivery_list dl ON d.delivery_id = dl.delivery_id " +
                    "JOIN Items i ON dl.item_id = i.item_id " +
                    "WHERE i.category = :category ",
            nativeQuery = true)
    List<Supplier> findSuppliersByItemCategory(@Param("category") String category);

    @Query(value =
            "SELECT COUNT(s), i.category " +
                    "FROM Supplier s " +
                    "JOIN Delivery d ON s.supplier_id = d.supplier_id " +
                    "JOIN delivery_list dl ON d.delivery_id = dl.delivery_id " +
                    "JOIN Items i ON dl.item_id = i.item_id " +
                    "WHERE i.category = :category " +
                    "GROUP BY i.category",
            nativeQuery = true)
    List<Supplier> findSuppliersCountByItemCategory(@Param("category") String category);

}
