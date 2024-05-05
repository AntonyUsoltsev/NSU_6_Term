package ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.sql.Timestamp;

@Data
@AllArgsConstructor
public class DefectItemsDto {
    private String itemName;
    private Integer defectAmount;
    private Timestamp deliveryDate;
    private String supplierName;
}
