package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Transaction;

import java.util.List;

@Data
@AllArgsConstructor
public class CashierDto {

    private Long cashierId;

    private String name;

    private String secondName;

    private List<Transaction> transactions;
}