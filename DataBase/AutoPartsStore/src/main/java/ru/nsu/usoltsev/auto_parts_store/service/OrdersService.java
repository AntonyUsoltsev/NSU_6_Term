package ru.nsu.usoltsev.auto_parts_store.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.exception.ResourceNotFoundException;
import ru.nsu.usoltsev.auto_parts_store.model.dto.OrdersDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Orders;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.OrdersMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.OrdersRepository;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class OrdersService {
    @Autowired
    private OrdersRepository ordersRepository;

    public OrdersDto saveOrder(OrdersDto ordersDto) {
        Orders customer = OrdersMapper.INSTANCE.fromDto(ordersDto);
        Orders savedItem = ordersRepository.saveAndFlush(customer);
        return OrdersMapper.INSTANCE.toDto(savedItem);
    }

    public OrdersDto getOrderById(Long id) {
        return OrdersMapper.INSTANCE.toDto(ordersRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Order is not found by id: " + id)));
    }

    public List<OrdersDto> getOrders() {
        return ordersRepository.findAll()
                .stream()
                .map(OrdersMapper.INSTANCE::toDto)
                .collect(Collectors.toList());
    }

}
