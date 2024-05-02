package ru.nsu.usoltsev.auto_parts_store.model.entity;

import jakarta.persistence.*;
import lombok.EqualsAndHashCode;

import java.io.Serializable;

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
    private Long itemId;
    @Id
    private Long orderId;
    @Column
    private Long amount;

}