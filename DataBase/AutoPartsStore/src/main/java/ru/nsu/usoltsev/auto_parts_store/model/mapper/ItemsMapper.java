package ru.nsu.usoltsev.auto_parts_store.model.mapper;

import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;
import ru.nsu.usoltsev.auto_parts_store.model.dto.ItemsDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Item;

@Mapper
public interface ItemsMapper {
    ItemsMapper INSTANCE = Mappers.getMapper(ItemsMapper.class);

    ItemsDto toDto(Item item);

    Item fromDto(ItemsDto itemsDto);
}
