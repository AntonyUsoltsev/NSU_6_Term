package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Supplier;

import java.sql.Timestamp;

@Data
@AllArgsConstructor
public class DeliveryDto {

    private Long deliveryId;

    private Supplier supplier;

    private Timestamp deliveryDate;
}
