package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ItemDto {

    private String name;

    private Long categoryId;

    private Integer amount;

    private Integer defectAmount;

    private Integer price;

    private Integer cellNumber;
}
