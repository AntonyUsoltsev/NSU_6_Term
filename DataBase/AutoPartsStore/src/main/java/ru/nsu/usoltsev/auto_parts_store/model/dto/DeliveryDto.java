package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Items;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Supplier;

import java.sql.Timestamp;
import java.util.List;

@Data
@AllArgsConstructor
public class DeliveryDto {

    private Long deliveryId;

    private Supplier supplier;

    private Timestamp deliveryDate;

    private List<Items> items;
}