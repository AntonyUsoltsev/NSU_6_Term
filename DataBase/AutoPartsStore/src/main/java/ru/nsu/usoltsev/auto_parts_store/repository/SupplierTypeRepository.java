package ru.nsu.usoltsev.auto_parts_store.repository;

import jakarta.transaction.Transactional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.SupplierType;

public interface SupplierTypeRepository extends JpaRepository<SupplierType, Long> {

    @Modifying
    @Transactional
    @Query("DELETE FROM SupplierType s WHERE s.typeId = :id")
    void deleteById(@Param("id") Long id);

    @Modifying
    @Transactional
    @Query(value = "INSERT INTO supplier_type (type_id, type_name) VALUES (default, :supplierTypeName)", nativeQuery = true)
    void addSupplierType(@Param("supplierTypeName") String supplierTypeName);

    @Modifying
    @Transactional
    @Query("UPDATE SupplierType s SET s.typeName = :newTypeName WHERE s.typeId = :id")
    void updateTypeNameById(@Param("id") Long id, @Param("newTypeName") String newTypeName);


    SupplierType findByTypeName(String typeName);
}
