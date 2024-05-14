package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierTypeDto;
import ru.nsu.usoltsev.auto_parts_store.service.CrudService;

@RestController
@RequestMapping("api/supplierType")
@CrossOrigin
@Slf4j
public class SupplierTypeController extends CrudController<SupplierTypeDto> {

    public SupplierTypeController(CrudService<SupplierTypeDto> crudService) {
        super(crudService);
    }
}
