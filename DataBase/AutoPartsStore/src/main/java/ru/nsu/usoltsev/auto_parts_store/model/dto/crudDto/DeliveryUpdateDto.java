package ru.nsu.usoltsev.auto_parts_store.model.dto.crudDto;

import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.dto.DeliveryDto;

import java.sql.Timestamp;
import java.util.List;

@Data
@AllArgsConstructor
public class DeliveryUpdateDto {
    @Data
    @AllArgsConstructor
    public static class ItemDeliveryDto {
        private Long itemId;
        private Long purchasePrice;
    }

    private Long supplierId;

    private Timestamp deliveryDate;

    private List<DeliveryDto.ItemDeliveryDto> itemsDelivery;
}
