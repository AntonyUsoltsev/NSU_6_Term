package ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CashierDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CustomerDto;

import java.sql.Timestamp;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CashReportDto {
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class TransactionItemList {
        private String itemName;
        private Long amount;
        private String supplierName;
        private Long price;
    }

    private Timestamp transactionDate;
    private String transactionType;
    private Integer fullPrice;
    private CashierDto cashier;
    private CustomerDto customer;
    private List<TransactionItemList> itemList;
}
