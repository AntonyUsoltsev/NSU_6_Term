package ru.nsu.usoltsev.auto_parts_store.controllers;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.nsu.usoltsev.auto_parts_store.model.dto.DeliveryDto;
import ru.nsu.usoltsev.auto_parts_store.service.CrudService;
import ru.nsu.usoltsev.auto_parts_store.service.DeliveryService;

@CrossOrigin
@RestController
@RequestMapping("api/delivery")
public class DeliveryController extends CrudController<DeliveryDto> {
    public DeliveryController(DeliveryService deliveryService) {
        super(deliveryService);
    }
}
