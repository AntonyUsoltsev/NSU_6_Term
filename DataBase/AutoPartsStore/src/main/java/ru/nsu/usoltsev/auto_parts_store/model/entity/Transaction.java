package ru.nsu.usoltsev.auto_parts_store.model.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.*;

import java.sql.Timestamp;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder

@Entity
@Table(name = "transaction")
public class Transaction {

    @Id
    @Column(name = "transaction_id", nullable = false)
    private Long transactionId;

    @Column(name = "order_id", nullable = false, unique = true)
    private Long orderId;

    @Column(name = "cashier_id", nullable = false)
    private Long cashierId;

    @Column(name = "type_id", nullable = false)
    private Long typeId;

    @Column(name = "date", nullable = false)
    private Timestamp transactionDate;
}

