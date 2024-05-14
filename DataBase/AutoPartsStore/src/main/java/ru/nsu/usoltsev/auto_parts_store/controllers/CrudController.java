package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import ru.nsu.usoltsev.auto_parts_store.service.CrudService;

import java.util.List;

@Slf4j
public abstract class CrudController<D> {
    private final CrudService<D> crudService;

    public CrudController(CrudService<D> crudService) {
        this.crudService = crudService;
    }

    @GetMapping("/all")
    public ResponseEntity<List<D>> getAll() {
        return ResponseEntity.ok(crudService.getAll());
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<String> delete(@PathVariable("id") Long id) {
        log.info("Delete row with id: {}", id);
        crudService.delete(id);
        return ResponseEntity.ok("Object deleted");
    }

    @PostMapping
    public ResponseEntity<Object> add(@RequestBody D dto) {
        crudService.add(dto);
        return ResponseEntity.ok().build();
    }

    @PatchMapping("/{id}")
    public ResponseEntity<Integer> update(@PathVariable("id") Long id, @RequestBody D dto) {
        log.info("Update row with id: {}, new value = {}", id, dto);
        crudService.update(id, dto);
        return ResponseEntity.ok().build();
    }
}

