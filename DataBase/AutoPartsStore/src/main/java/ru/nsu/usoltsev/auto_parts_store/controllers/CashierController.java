package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CashierDto;
import ru.nsu.usoltsev.auto_parts_store.service.CashierService;
import ru.nsu.usoltsev.auto_parts_store.service.CrudService;

@CrossOrigin
@RestController
@RequestMapping("api/cashiers")
public class CashierController extends CrudController<CashierDto> {

    public CashierController(CashierService cashierService) {
        super(cashierService);
    }
}
