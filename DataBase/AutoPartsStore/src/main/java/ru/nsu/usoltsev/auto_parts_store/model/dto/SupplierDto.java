package ru.nsu.usoltsev.auto_parts_store.model.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Delivery;

import java.util.List;

@Data
@AllArgsConstructor
public class SupplierDto {

    private Long supplierId;

    @NotBlank(message = "Name cannot be blank")
    private String name;

    private String documents;

    private String type;

    private Boolean garanty;

    private List<Delivery> deliveries;
}
