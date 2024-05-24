package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.sql.Timestamp;
import java.util.List;

@Data
@AllArgsConstructor
public class OrdersDto {
    @Data
    @AllArgsConstructor
    public static class ItemOrderDto {
        private ItemDto item;
        private Long amount;
    }

    private Long orderId;

    private Long customerId;

    private Timestamp orderDate;

    private Integer fullPrice;

    private CustomerDto customer;

    private List<OrdersDto.ItemOrderDto> itemsOrders;
}
