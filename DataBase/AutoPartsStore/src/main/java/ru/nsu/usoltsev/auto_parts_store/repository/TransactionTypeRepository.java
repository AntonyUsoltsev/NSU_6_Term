package ru.nsu.usoltsev.auto_parts_store.repository;

import jakarta.transaction.Transactional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.TransactionType;

public interface TransactionTypeRepository extends JpaRepository<TransactionType, Long> {
    @Modifying
    @Transactional
    @Query("DELETE FROM TransactionType t WHERE t.typeId = :id")
    void deleteById(@Param("id") Long id);

    @Modifying
    @Transactional
    @Query(value = "INSERT INTO transaction_type (type_id, type_name) VALUES (default, :typeName)", nativeQuery = true)
    void addTransactionType(@Param("typeName") String typeName);

    @Modifying
    @Transactional
    @Query("UPDATE TransactionType t  SET t.typeName = :newTypeName WHERE t.typeId = :id")
    void updateTypeNameById(@Param("id") Long id, @Param("newTypeName") String newTypeName);

    TransactionType findByTypeName(String typeName);

    @Query("SELECT tt " +
            "FROM  TransactionType tt " +
            "WHERE tt.typeId = :id")
    TransactionType findByTypeId(@Param("id") Long id);
}
