package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Customer;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Items;

import java.sql.Timestamp;
import java.util.List;

@Data
@AllArgsConstructor
public class OrdersDto {

    private Long orderId;

    private Customer customer;

    private Timestamp orderDate;

    private Integer fullPrice;

    private List<Items> items;
}
