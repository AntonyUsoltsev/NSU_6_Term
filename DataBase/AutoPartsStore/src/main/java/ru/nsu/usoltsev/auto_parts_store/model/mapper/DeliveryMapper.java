package ru.nsu.usoltsev.auto_parts_store.model.mapper;

import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;
import ru.nsu.usoltsev.auto_parts_store.model.dto.DeliveryDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Delivery;

@Mapper
public interface DeliveryMapper {
    DeliveryMapper INSTANCE = Mappers.getMapper(DeliveryMapper.class);

    DeliveryDto toDto(Delivery delivery);

    Delivery fromDto(DeliveryDto deliveryDto);
}
