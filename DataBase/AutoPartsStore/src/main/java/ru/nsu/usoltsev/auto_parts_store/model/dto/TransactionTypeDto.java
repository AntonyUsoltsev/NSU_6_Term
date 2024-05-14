package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class TransactionTypeDto {
    private Long typeId;
    private String typeName;
}
