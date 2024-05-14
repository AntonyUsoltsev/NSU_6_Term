package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.nsu.usoltsev.auto_parts_store.model.dto.TransactionTypeDto;
import ru.nsu.usoltsev.auto_parts_store.service.TransactionTypeService;

@RestController
@RequestMapping("api/transactionType")
@CrossOrigin
@Slf4j
public class TransactionTypeController extends CrudController<TransactionTypeDto> {
    public TransactionTypeController(TransactionTypeService transactionTypeService) {
        super(transactionTypeService);
    }
}
