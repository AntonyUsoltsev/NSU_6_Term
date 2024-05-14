package ru.nsu.usoltsev.auto_parts_store.repository;

import jakarta.transaction.Transactional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.ItemCategory;

public interface ItemCategoryRepository extends JpaRepository<ItemCategory, Long> {

    @Modifying
    @Transactional
    @Query("DELETE FROM ItemCategory ic WHERE ic.categoryId = :id")
    void deleteById(@Param("id") Long id);

    @Modifying
    @Transactional
    @Query(value = "INSERT INTO item_category (category_id, category_name) VALUES (default, :itemCategory)", nativeQuery = true)
    void addItemCategory(@Param("itemCategory") String itemCategory);

    @Modifying
    @Transactional
    @Query("UPDATE ItemCategory ic SET ic.categoryName = :newTypeName WHERE ic.categoryId = :id")
    void updateTypeNameById(@Param("id") Long id, @Param("newTypeName") String newTypeName);
}
