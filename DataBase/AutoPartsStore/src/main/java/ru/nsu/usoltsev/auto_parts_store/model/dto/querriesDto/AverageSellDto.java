package ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class AverageSellDto {
    private String typeName;
    private Long sellAmount;
    private Double averageSell;
}
