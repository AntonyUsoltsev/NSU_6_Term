package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import ru.nsu.usoltsev.auto_parts_store.model.Params;
import ru.nsu.usoltsev.auto_parts_store.model.dto.ItemDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.*;
import ru.nsu.usoltsev.auto_parts_store.service.ItemService;

import java.util.List;

@RestController
@RequestMapping("api/items")
@CrossOrigin
@Slf4j
public class ItemsController extends CrudController<ItemDto> {
    private final ItemService itemsRepository;

    public ItemsController(ItemService itemsRepository) {
        super(itemsRepository);
        this.itemsRepository = itemsRepository;
    }

    @GetMapping("/{id}")
    public ResponseEntity<ItemDto> getItem(@PathVariable String id) {
        return ResponseEntity.ok(itemsRepository.getItemById(Long.valueOf(id)));
    }

//    @GetMapping("/all")
//    public ResponseEntity<List<ItemDto>> getItems() {
//        return ResponseEntity.ok(itemsRepository.getItems());
//    }

    @GetMapping("/top")
    public ResponseEntity<List<TopTenItemsDto>> getTopTen() {
        return ResponseEntity.ok(itemsRepository.getTopTen());
    }

    @GetMapping("/info")
    public ResponseEntity<List<ItemInfoDto>> getItemsInfo() {
        return ResponseEntity.ok(itemsRepository.getItemsInfo());
    }

    @GetMapping("/catalog")
    public ResponseEntity<List<ItemCatalogDto>> getItemsCatalog() {
        log.info("Get catalog");
        return ResponseEntity.ok(itemsRepository.getItemsCatalog());
    }

    @GetMapping("/deliveryPrice")
    public ResponseEntity<List<ItemDeliveryPriceDto>> getItemDeliveryPrice() {
        return ResponseEntity.ok(itemsRepository.getItemDeliveryPrice());
    }

    @GetMapping("/defect")
    public ResponseEntity<List<DefectItemsDto>> getDefectItems(@RequestParam("from") String fromDate,
                                                               @RequestParam("to") String toDate) {
        List<DefectItemsDto> items = itemsRepository.getDefectItems(fromDate, toDate);
        return ResponseEntity.ok(items);
    }

    @GetMapping()
    public ResponseEntity<List<ItemDto>> getItemsByCategory(@RequestParam("category") String category) {
        List<ItemDto> items = itemsRepository.getItemsByCategory(category);
        return ResponseEntity.ok(items);
    }

    @GetMapping("/storeCapacity")
    public ResponseEntity<Integer> getStoreCapacity() {
        Integer capacity = itemsRepository.getStoreCapacity();
        return ResponseEntity.ok(Params.storeCapacity - capacity);
    }


//    @PostMapping()
//    public ResponseEntity<ItemDto> createItem(@RequestBody ItemDto itemDto) {
//        return new ResponseEntity<>(itemsRepository.saveItem(itemDto), HttpStatus.CREATED);
//    }

//
//    @GetMapping("/{id}")
//    public ResponseEntity<ItemDto> getItemById(@PathVariable("id") Long id) {
//        ItemDto item = itemsRepository.getItemById(id);
//        if (item != null) {
//            return ResponseEntity.ok(item);
//        } else {
//            return ResponseEntity.notFound().build();
//        }
//    }

}
