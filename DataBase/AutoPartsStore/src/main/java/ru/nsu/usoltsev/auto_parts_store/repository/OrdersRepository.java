package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Orders;

public interface OrdersRepository extends JpaRepository<Orders, Long> {
}