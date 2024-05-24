package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Supplier;

import java.sql.Timestamp;
import java.util.List;

@Data
@AllArgsConstructor
public class DeliveryDto {

    @Data
    @AllArgsConstructor
    public static class ItemDeliveryDto {
        private ItemDto item;
        private Long purchasePrice;
        private Long amount;
    }

    private Long deliveryId;

    private Long supplierId;

    private Timestamp deliveryDate;

    private SupplierDto supplier;

    private List<ItemDeliveryDto> itemsDelivery;
}
