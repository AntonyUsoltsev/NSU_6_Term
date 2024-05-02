package ru.nsu.usoltsev.auto_parts_store.model.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder

@Entity
@Table(name = "supplier_type")
public class SupplierType {
    @Id
    @Column(name = "type_id", nullable = false)
    private Long typeId;

    @Column(name = "type_name", nullable = false)
    private String typeName;
}
