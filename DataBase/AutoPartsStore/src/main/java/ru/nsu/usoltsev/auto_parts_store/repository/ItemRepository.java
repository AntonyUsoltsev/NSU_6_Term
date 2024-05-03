package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.ItemInfoDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Item;

import java.util.List;

public interface ItemRepository extends JpaRepository<Item, Long> {
    @Query( "SELECT i " +
            "FROM Item i " +
            "LEFT JOIN ItemCategory ic on i.categoryId = ic.categoryId " +
            "WHERE ic.categoryName = :category")
    List<Item> findByCategory(@Param("category") String category);

    @Query( "SELECT i.name, i.amount, i.cellNumber " +
            "FROM Item i ")
    List<Object[]> findAllItemsInfo();

    @Query("SELECT i.name AS item_name, SUM(ol.amount) AS total_sold " +
            "FROM Item i " +
            "JOIN OrderList ol ON i.itemId = ol.itemId " +
            "GROUP BY i.name " +
            "ORDER BY total_sold DESC " +
            "LIMIT 10")
    List<Object[]> getTopTenSoldDetails();
}
