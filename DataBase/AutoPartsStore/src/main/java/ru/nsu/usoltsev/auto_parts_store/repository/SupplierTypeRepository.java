package ru.nsu.usoltsev.auto_parts_store.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.nsu.usoltsev.auto_parts_store.model.entity.SupplierType;

public interface SupplierTypeRepository extends JpaRepository<SupplierType, Long> {
}
