package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Customer;

public interface CustomerRepository extends JpaRepository<Customer, Long> {
}
