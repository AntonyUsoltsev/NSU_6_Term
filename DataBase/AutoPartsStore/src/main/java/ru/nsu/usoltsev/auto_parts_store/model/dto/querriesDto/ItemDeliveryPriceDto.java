package ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.sql.Timestamp;
import java.util.List;

@Data
@AllArgsConstructor
public class ItemDeliveryPriceDto {
    @Data
    @AllArgsConstructor
    public static class SupplierDeliveryPrice {
        private String supplierName;
        private Integer price;
        private Timestamp deliveryDate;
    }

    private String name;
    private List<SupplierDeliveryPrice> supplierDeliveryPriceList;
}
