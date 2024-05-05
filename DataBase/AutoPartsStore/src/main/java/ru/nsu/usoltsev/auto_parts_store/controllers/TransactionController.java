package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.AllArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.CashReportDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.SellingSpeedDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.TransactionInfoDto;
import ru.nsu.usoltsev.auto_parts_store.service.TransactionService;

import java.util.List;

@RestController
@RequestMapping("api/transactions")
@AllArgsConstructor
public class TransactionController {
    @Autowired
    private TransactionService transactionService;

    @GetMapping("/realised")
    public ResponseEntity<List<TransactionInfoDto>> getRealisedItemsByDay(@RequestParam("date") String date) {
        return ResponseEntity.ok(transactionService.getTransactionInfoByDay(date));
    }

    @GetMapping("/sellSpeed")
    public ResponseEntity<List<SellingSpeedDto>> getSellingSpeed() {
        return ResponseEntity.ok(transactionService.getSellingSpeed());
    }

    @GetMapping("/cashReport")
    public ResponseEntity<List<CashReportDto>> getCashReport(@RequestParam("from") String fromDate,
                                                             @RequestParam("to") String toDate) {
        return ResponseEntity.ok(transactionService.getCashReport(fromDate, toDate));
    }
}
