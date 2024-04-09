package ru.nsu.usoltsev.auto_parts_store.model.mapper;

import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;
import ru.nsu.usoltsev.auto_parts_store.model.dto.OrdersDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Orders;

@Mapper
public interface OrdersMapper {
    OrdersMapper INSTANCE = Mappers.getMapper(OrdersMapper.class);

    OrdersDto toDto(Orders orders);

    Orders fromDto(OrdersDto ordersDto);
}
