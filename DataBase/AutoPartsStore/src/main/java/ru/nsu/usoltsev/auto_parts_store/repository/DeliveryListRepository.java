package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.nsu.usoltsev.auto_parts_store.model.entity.DeliveryList;

import java.util.List;

public interface DeliveryListRepository extends JpaRepository<DeliveryList, DeliveryList.DeliveryListKey>  {

    @Query("SELECT dl " +
            "FROM DeliveryList dl " +
            "WHERE dl.deliveryId = :deliveryId")
    List<DeliveryList> findDeliveriesByDeliveryId(@Param("deliveryId") Long deliveryId);
}
