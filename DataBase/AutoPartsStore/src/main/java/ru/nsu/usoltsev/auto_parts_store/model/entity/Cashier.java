package ru.nsu.usoltsev.auto_parts_store.model.entity;

import jakarta.persistence.*;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder

@Entity
@Table(name = "cashier")
public class Cashier {

    @Id
    @Column(name = "cashier_id", nullable = false)
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long cashierId;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "second_name", nullable = false)
    private String secondName;
}
