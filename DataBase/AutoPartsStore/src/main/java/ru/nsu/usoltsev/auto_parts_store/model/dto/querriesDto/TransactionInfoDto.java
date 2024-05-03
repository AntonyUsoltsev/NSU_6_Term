package ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class TransactionInfoDto {
    private String name;
    private Long amount;
    private Long price;
}
