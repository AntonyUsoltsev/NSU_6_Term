package ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto;

import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierDto;

import java.util.List;

@Data
@AllArgsConstructor
public class SupplierByTypeDto {

    private List<SupplierDto> suppliers;

    private Integer count;
}
