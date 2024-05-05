package ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.sql.Timestamp;
import java.time.Duration;

@Data
@AllArgsConstructor
public class SellingSpeedDto {
    private String name;
    private Timestamp transactionDate;
    private Timestamp deliveryDate;
    private Long timeDiff;
}
