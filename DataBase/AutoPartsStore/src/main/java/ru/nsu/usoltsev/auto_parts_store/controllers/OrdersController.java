package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.nsu.usoltsev.auto_parts_store.model.dto.OrdersDto;
import ru.nsu.usoltsev.auto_parts_store.service.OrdersService;

import java.util.List;

@RestController
@RequestMapping("api/orders")
@AllArgsConstructor
public class OrdersController {

    private OrdersService ordersService;

    @GetMapping("/all")
    public ResponseEntity<List<OrdersDto>> getOrders() {
        return ResponseEntity.ok(ordersService.getOrders());
    }

}
