package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import ru.nsu.usoltsev.auto_parts_store.model.dto.TransactionDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.AverageSellDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.CashReportDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.SellingSpeedDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.TransactionInfoDto;
import ru.nsu.usoltsev.auto_parts_store.service.TransactionService;

import java.util.List;

@Slf4j
@RestController
@CrossOrigin
@RequestMapping("api/transactions")
public class TransactionController extends CrudController<TransactionDto> {
    @Autowired
    private TransactionService transactionService;

    public TransactionController(TransactionService transactionService) {
        super(transactionService);
        this.transactionService = transactionService;
    }

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
        log.info("from date {}, to date {}", fromDate, toDate);
        return ResponseEntity.ok(transactionService.getCashReport(fromDate, toDate));
    }

    @GetMapping("/averageSell")
    public ResponseEntity<List<AverageSellDto>> getAverageSell() {
        return ResponseEntity.ok(transactionService.getAverageSell());
    }

}
