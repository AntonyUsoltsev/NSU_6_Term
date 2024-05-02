package ru.nsu.usoltsev.auto_parts_store.model.entity;

import jakarta.persistence.*;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder

@Entity
@Table(name = "item")
public class Item {

    @Id
    @Column(name = "item_id", nullable = false)
    private Long itemId;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "category_id", nullable = false)
    private Long categoryId;

    @Column(name = "amount", nullable = false)
    private Integer amount;

    @Column(name = "defect_amount", nullable = false)
    private Integer defectAmount;

    @Column(name = "price", nullable = false)
    private Integer price;

    @Column(name = "cell_number", nullable = false)
    private Integer cellNumber;

}
