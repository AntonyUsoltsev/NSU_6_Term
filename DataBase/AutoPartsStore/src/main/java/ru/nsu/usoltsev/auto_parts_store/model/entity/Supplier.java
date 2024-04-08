package ru.nsu.usoltsev.auto_parts_store.model.entity;

import jakarta.persistence.*;
import lombok.*;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder

@Entity
@Table(name = "supplier")
public class Supplier {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    @Column(name = "supplier_id", nullable = false)
    private Long supplierId;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "documents", nullable = false)
    private String documents;

    @Column(name = "type", nullable = false)
    private String type;

    @Column(name = "garanty", nullable = false)
    private Boolean garanty;

    @OneToMany(mappedBy = "supplier", cascade = {CascadeType.PERSIST, CascadeType.MERGE, CascadeType.REFRESH} )
    private List<Delivery> deliveries;
}
