package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
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
        return ResponseEntity.ok(supplierTypeService.getSupplierTypes());
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteSupplierType(@PathVariable("id") Long id) {
        log.info("Delete supplier type with id: {}", id);
        supplierTypeService.deleteSupplierType(id);
        return ResponseEntity.ok("Object deleted");
    }

    @PostMapping
    public ResponseEntity<Object> addSupplierType(@RequestBody SupplierTypeDto supplierTypeDto) {
        supplierTypeService.addSupplierType(supplierTypeDto);
        return ResponseEntity.ok().build();
    }

    @PatchMapping("/{id}")
    public ResponseEntity<Integer> updateSupplierType(@PathVariable("id") Long id, @RequestBody SupplierTypeDto supplierTypeDto) {
        log.info("Update supplier type with id: {}, new value = {}", id, supplierTypeDto);
        supplierTypeService.updateSupplierType(id, supplierTypeDto);
        return ResponseEntity.ok().build();
    }

}
