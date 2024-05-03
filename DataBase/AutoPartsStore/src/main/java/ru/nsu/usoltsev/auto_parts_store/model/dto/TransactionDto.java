package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Cashier;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Orders;

import java.sql.Timestamp;

@Data
@AllArgsConstructor
public class TransactionDto {
    private Long transactionId;

    private Orders orders;

    private Cashier cashier;

    private Long typeId;

    private Timestamp transactionDate;
}
