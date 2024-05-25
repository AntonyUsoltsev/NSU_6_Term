package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Cashier;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Orders;

import java.sql.Timestamp;

@Data
@AllArgsConstructor
public class TransactionDto {
    private Long transactionId;

//    private Long orderId;
//
//    private Long cashierId;
//
//    private Long typeId;

    private Timestamp transactionDate;

    private OrdersDto orders;

    private CashierDto cashier;

    private TransactionTypeDto transactionTypeDto;
}
