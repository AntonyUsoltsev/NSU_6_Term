package ru.nsu.usoltsev.auto_parts_store.service;

import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.exception.ResourceNotFoundException;
import ru.nsu.usoltsev.auto_parts_store.model.dto.OrdersDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Customer;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Item;
import ru.nsu.usoltsev.auto_parts_store.model.entity.OrderList;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Orders;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.CustomerMapper;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.ItemMapper;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.OrdersMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.CustomerRepository;
import ru.nsu.usoltsev.auto_parts_store.repository.ItemRepository;
import ru.nsu.usoltsev.auto_parts_store.repository.OrderListRepository;
import ru.nsu.usoltsev.auto_parts_store.repository.OrdersRepository;

import java.util.ArrayList;
import java.util.List;

@Service
@Transactional
public class OrdersService implements CrudService<OrdersDto> {
    @Autowired
    private OrdersRepository ordersRepository;

    @Autowired
    private ItemRepository itemRepository;

    @Autowired
    private CustomerRepository customerRepository;

    @Autowired
    private OrderListRepository orderListRepository;

    public OrdersDto saveOrder(OrdersDto ordersDto) {
        Orders orders = OrdersMapper.INSTANCE.fromDto(ordersDto);
        Orders savedItem = ordersRepository.saveAndFlush(orders);
        return OrdersMapper.INSTANCE.toDto(savedItem);
    }

    public OrdersDto getOrderById(Long id) {
        return OrdersMapper.INSTANCE.toDto(ordersRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Order is not found by id: " + id)));
    }

    @Override
    public List<OrdersDto> getAll() {
        List<OrdersDto> ordersDtos = new ArrayList<>();
        List<Orders> orders = ordersRepository.findAll();
        for (Orders order : orders) {
            Customer customer = customerRepository.findById(order.getCustomerId()).orElseThrow();
            List<OrderList> orderLists = orderListRepository.findOrdersByOrderId(order.getOrderId());
            List<OrdersDto.ItemOrderDto> items = new ArrayList<>();
            for (OrderList orderList : orderLists) {
                Item item = itemRepository.findById(orderList.getItemId()).orElseThrow();
                items.add(new OrdersDto.ItemOrderDto(
                        ItemMapper.INSTANCE.toDto(item),
                        orderList.getAmount()));
            }
            OrdersDto orderDto = OrdersMapper.INSTANCE.toDto(order);
            orderDto.setCustomer(CustomerMapper.INSTANCE.toDto(customer));
            orderDto.setItemsOrders(items);
            ordersDtos.add(orderDto);
        }
        return ordersDtos;
    }


    @Override
    public void delete(Long id) {

    }

    @Override
    public OrdersDto add(OrdersDto dto) {
        for (OrdersDto.ItemOrderDto itemsOrder : dto.getItemsOrders()) {
            Item item = itemRepository.findById(itemsOrder.getItem().getItemId()).orElseThrow();
            if (item.getAmount() - itemsOrder.getAmount() < 0) {
                throw new RuntimeException("Too few details to add to new order");
            }
        }
        Orders orders = new Orders(dto.getOrderId(), dto.getCustomerId(), dto.getOrderDate(), 0);
        Orders savedOrders = ordersRepository.saveAndFlush(orders);

        for (OrdersDto.ItemOrderDto itemsOrder : dto.getItemsOrders()) {
            Item item = itemRepository.findById(itemsOrder.getItem().getItemId()).orElseThrow();
            item.setAmount((int) (item.getAmount() - itemsOrder.getAmount()));
            itemRepository.saveAndFlush(item);
            OrderList orderList = new OrderList(
                    itemsOrder.getItem().getItemId(),
                    savedOrders.getOrderId(),
                    itemsOrder.getAmount()
            );
            orderListRepository.saveAndFlush(orderList);
        }
        updateFullPrice();
        return OrdersMapper.INSTANCE.toDto(savedOrders);
    }

    @Override
    public void update(Long id, OrdersDto dto) {
        List<OrderList> existingOrderLists = orderListRepository.findOrdersByOrderId(id);

        for (OrdersDto.ItemOrderDto itemsOrder : dto.getItemsOrders()) {
            Item item = itemRepository.findById(itemsOrder.getItem().getItemId()).orElseThrow();
            OrderList orderItemAmountCheck = existingOrderLists.stream().filter(orderList -> orderList.getOrderId().equals(id)
                    && orderList.getItemId().equals(item.getItemId())).findFirst().orElseThrow();
            if (item.getAmount() + orderItemAmountCheck.getAmount() - itemsOrder.getAmount() < 0) {
                throw new RuntimeException("Too few details to add to order");
            }
        }

        Orders orders = ordersRepository.findById(id).orElseThrow();
        orders.setCustomerId(dto.getCustomerId());
        orders.setOrderDate(dto.getOrderDate());
        ordersRepository.saveAndFlush(orders);

        orderListRepository.deleteAll(existingOrderLists);
        orderListRepository.flush();

        for (OrdersDto.ItemOrderDto itemsOrder : dto.getItemsOrders()) {
            Item item = itemRepository.findById(itemsOrder.getItem().getItemId()).orElseThrow();
            OrderList orderItemAmountCheck = existingOrderLists.stream().filter(orderList -> orderList.getOrderId().equals(id)
                    && orderList.getItemId().equals(item.getItemId())).findFirst().orElseThrow();
            item.setAmount((int) (item.getAmount() + orderItemAmountCheck.getAmount() - itemsOrder.getAmount()));
            itemRepository.saveAndFlush(item);

            OrderList orderList = new OrderList(
                    id,
                    itemsOrder.getItem().getItemId(),
                    itemsOrder.getAmount()
            );
            orderListRepository.saveAndFlush(orderList);
        }
        updateFullPrice();
    }

    public void updateFullPrice() {
        List<Orders> orders = ordersRepository.findAll();
        for (Orders order : orders) {
            Integer newFullPrice = orderListRepository.getOrderFullPrice(order.getOrderId());
            order.setFullPrice(newFullPrice);
            ordersRepository.saveAndFlush(order);
        }
    }
}
