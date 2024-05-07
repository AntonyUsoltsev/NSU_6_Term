package ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

@Data
@AllArgsConstructor

public class ItemCatalogDto {
    @Data
    @AllArgsConstructor
    public static class SupplierItemInfo {
        private Integer amount;

        private Integer price;

        private String supplierName;
    }

    private String itemName;

    private String categoryName;

    private List<SupplierItemInfo> supplierItemInfos;

}

