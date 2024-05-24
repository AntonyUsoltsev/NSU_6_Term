package ru.nsu.usoltsev.auto_parts_store.controllers;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.nsu.usoltsev.auto_parts_store.model.dto.OrdersDto;
import ru.nsu.usoltsev.auto_parts_store.service.OrdersService;

@RestController
@CrossOrigin
@RequestMapping("api/orders")
public class OrdersController extends CrudController<OrdersDto> {

    public OrdersController(OrdersService ordersService) {
        super(ordersService);
    }
}
