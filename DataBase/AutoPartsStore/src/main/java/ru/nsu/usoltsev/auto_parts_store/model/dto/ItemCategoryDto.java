package ru.nsu.usoltsev.auto_parts_store.model.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor

public class ItemCategoryDto {
    @JsonProperty(required = false)
    private Long categoryId;
    private String categoryName;
}
