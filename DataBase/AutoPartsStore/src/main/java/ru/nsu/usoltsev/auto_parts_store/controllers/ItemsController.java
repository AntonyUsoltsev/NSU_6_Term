package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.AllArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import ru.nsu.usoltsev.auto_parts_store.model.dto.ItemsDto;
import ru.nsu.usoltsev.auto_parts_store.service.ItemsService;

import java.util.List;

@RestController
@RequestMapping("api/items")
@AllArgsConstructor
public class ItemsController {
    private ItemsService itemsRepository;

    @GetMapping("/{id}")
    public ResponseEntity<ItemsDto> getItem(@PathVariable String id) {
        return ResponseEntity.ok(itemsRepository.getItemById(Long.valueOf(id)));
    }

    @GetMapping("/all")
    public ResponseEntity<List<ItemsDto>> getItems() {
        return ResponseEntity.ok(itemsRepository.getItems());
    }
//
//    @GetMapping("/{id}")
//    public ResponseEntity<ItemsDto> getItemById(@PathVariable("id") Long id) {
//        ItemsDto item = itemsRepository.getItemById(id);
//        if (item != null) {
//            return ResponseEntity.ok(item);
//        } else {
//            return ResponseEntity.notFound().build();
//        }
//    }

    @GetMapping()
    public ResponseEntity<List<ItemsDto>> getItemsByCategory(@RequestParam("category") String category) {
        List<ItemsDto> items = itemsRepository.getItemsByCategory(category);
        return ResponseEntity.ok(items);
    }

    @PostMapping()
    public ResponseEntity<ItemsDto> createCustomer(@RequestBody ItemsDto itemDto) {
        return new ResponseEntity<>(itemsRepository.saveItem(itemDto), HttpStatus.CREATED);
    }
}
