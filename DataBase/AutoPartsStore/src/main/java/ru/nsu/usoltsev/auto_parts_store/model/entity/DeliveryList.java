package ru.nsu.usoltsev.auto_parts_store.model.entity;

import jakarta.persistence.*;
import lombok.EqualsAndHashCode;

import java.io.Serializable;

@Entity
@Table(name = "delivery_list")
@IdClass(DeliveryList.DeliveryListKey.class)
public class DeliveryList {

    @EqualsAndHashCode
    public static class DeliveryListKey implements Serializable {
        private Long itemId;
        private Long deliveryId;
    }

    @Id
    private Long itemId;
    @Id
    private Long deliveryId;
    @Column
    private Long amount;
}
