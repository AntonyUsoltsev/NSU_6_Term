package ru.nsu.usoltsev.auto_parts_store.model.mapper;

import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;
import ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierTypeDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.SupplierType;

@Mapper
public interface SupplierTypeMapper {
    SupplierTypeMapper INSTANCE = Mappers.getMapper(SupplierTypeMapper.class);

    SupplierTypeDto toDto(SupplierType customer);

    SupplierType fromDto(SupplierTypeDto customerDto);
}
