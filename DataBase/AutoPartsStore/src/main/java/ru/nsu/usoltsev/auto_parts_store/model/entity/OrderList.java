package ru.nsu.usoltsev.auto_parts_store.model.entity;

import jakarta.persistence.*;
import lombok.*;

import java.io.Serializable;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Entity
@Table(name = "order_list")
@IdClass(OrderList.OrderListKey.class)
public class OrderList {

    @EqualsAndHashCode
    public static class OrderListKey implements Serializable {
        private Long itemId;
        private Long orderId;
    }

    @Id
    @Column(name = "item_id")
    private Long itemId;

    @Id
    @Column(name = "order_id")
    private Long orderId;

    @Column(name = "amount")
    private Long amount;

}