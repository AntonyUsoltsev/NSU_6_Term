package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Cashier;

public interface CashierRepository extends JpaRepository<Cashier, Long> {
}
