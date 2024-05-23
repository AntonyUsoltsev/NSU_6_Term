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
@Table(name = "delivery_list")
@IdClass(DeliveryList.DeliveryListKey.class)
public class DeliveryList {

    @Id
    @Column(name = "delivery_id")
    private Long deliveryId;

    @Id
    @Column(name = "item_id")
    private Long itemId;

    @Column(name = "amount")
    private Long amount;

    @Column(name = "purchase_price")
    private Long purchasePrice;

    @EqualsAndHashCode
    public static class DeliveryListKey implements Serializable {
        private Long itemId;
        private Long deliveryId;
    }
}
