package ru.nsu.usoltsev.auto_parts_store.model.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class SupplierDto {

    private Long supplierId;

    @NotBlank(message = "Name cannot be blank")
    private String name;

    private String documents;

    private String typeName;

    private Boolean garanty;

}
