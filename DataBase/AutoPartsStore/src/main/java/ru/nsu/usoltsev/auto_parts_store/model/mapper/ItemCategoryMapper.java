package ru.nsu.usoltsev.auto_parts_store.model.mapper;

import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;
import ru.nsu.usoltsev.auto_parts_store.model.dto.ItemCategoryDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.ItemCategory;

@Mapper
public interface ItemCategoryMapper {
    ItemCategoryMapper INSTANCE = Mappers.getMapper(ItemCategoryMapper.class);

    ItemCategoryDto toDto(ItemCategory itemCategory);

    ItemCategory fromDto(ItemCategoryDto itemCategoryDto);
}
