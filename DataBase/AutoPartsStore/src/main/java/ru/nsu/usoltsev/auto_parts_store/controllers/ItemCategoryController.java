package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.nsu.usoltsev.auto_parts_store.model.dto.ItemCategoryDto;
import ru.nsu.usoltsev.auto_parts_store.service.ItemCategoryService;

@RestController
@RequestMapping("api/itemCategory")
@CrossOrigin
@Slf4j
public class ItemCategoryController extends CrudController<ItemCategoryDto> {

    public ItemCategoryController(ItemCategoryService Service) {
        super(Service);
    }

}
