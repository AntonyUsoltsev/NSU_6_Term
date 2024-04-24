package ru.nsu.usoltsev.auto_parts_store.controllers;


import lombok.AllArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierDto;
import ru.nsu.usoltsev.auto_parts_store.service.SupplierService;

import java.util.List;

@RestController
@RequestMapping("api/suppliers")
@AllArgsConstructor
public class SupplierController {

    private SupplierService supplierService;

    @GetMapping("/{id}")
    public ResponseEntity<SupplierDto> getSupplier(@PathVariable String id) {
        return ResponseEntity.ok(supplierService.getSupplierById(Long.valueOf(id)));
    }

    @GetMapping()
    public ResponseEntity<List<SupplierDto>> getSuppliers() {
        return ResponseEntity.ok(supplierService.getSuppliers());
    }

    @PostMapping()
    public ResponseEntity<SupplierDto> createSupplier(@RequestBody SupplierDto supplierDto) {
        return new ResponseEntity<>(supplierService.saveSupplier(supplierDto), HttpStatus.CREATED);
    }
}
