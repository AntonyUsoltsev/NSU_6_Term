package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Transaction;

public interface TransactionRepository extends JpaRepository<Transaction, Long> {
}
