package ru.nsu.usoltsev.auto_parts_store.model.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.query.Order;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder

@Entity
@Table(name = "items")
public class Items {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "item_id", nullable = false)
    private Long itemId;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "category", nullable = false)
    private String category;

    @Column(name = "amount", nullable = false)
    private Integer amount;

    @Column(name = "defect_amount", nullable = false)
    private Integer defectAmount;

    @Column(name = "price", nullable = false)
    private Integer price;

    @Column(name = "cell_number", nullable = false)
    private Integer cellNumber;

    @ManyToMany
    @JoinTable(
            name = "delivery_list",
            joinColumns = @JoinColumn(name = "item_id"),
            inverseJoinColumns = @JoinColumn(name = "delivery_id")
    )
    private List<Delivery> deliveries;

    @ManyToMany
    @JoinTable(
            name = "order_list",
            joinColumns = @JoinColumn(name = "item_id"),
            inverseJoinColumns = @JoinColumn(name = "order_id")
    )
    private List<Orders> orders;
}
