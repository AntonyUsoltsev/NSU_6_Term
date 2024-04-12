package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Items;

import java.util.List;

public interface ItemsRepository extends JpaRepository<Items, Long> {
    @Query("SELECT i " +
            "FROM Items i " +
            "WHERE i.category = :category")
    List<Items> findByCategory(@Param("category") String category);
}
