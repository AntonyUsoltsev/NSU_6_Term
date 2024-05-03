package ru.nsu.usoltsev.auto_parts_store.model.mapper;

import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;
import ru.nsu.usoltsev.auto_parts_store.model.dto.ItemDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Item;

@Mapper
public interface ItemMapper {
    ItemMapper INSTANCE = Mappers.getMapper(ItemMapper.class);

    ItemDto toDto(Item item);

    Item fromDto(ItemDto itemDto);
}
