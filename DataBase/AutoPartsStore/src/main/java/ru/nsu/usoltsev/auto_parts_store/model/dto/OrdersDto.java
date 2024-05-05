package ru.nsu.usoltsev.auto_parts_store.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Customer;

import java.sql.Timestamp;

@Data
@AllArgsConstructor
public class OrdersDto {

    private Long orderId;

    private CustomerDto customer;

    private Timestamp orderDate;

    private Integer fullPrice;
}
