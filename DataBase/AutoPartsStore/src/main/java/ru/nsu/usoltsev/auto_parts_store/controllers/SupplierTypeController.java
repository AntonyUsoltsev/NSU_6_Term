package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierTypeDto;
import ru.nsu.usoltsev.auto_parts_store.service.SupplierTypeService;

import java.util.List;

@RestController
@RequestMapping("api/supplierType")
@CrossOrigin
@AllArgsConstructor
@Slf4j
public class SupplierTypeController {
    private SupplierTypeService supplierTypeService;

    @GetMapping("/all")
    public ResponseEntity<List<SupplierTypeDto>> getSupplierTypes() {
        return ResponseEntity.ok(supplierTypeService.getTransactionTypes());
    }
}
