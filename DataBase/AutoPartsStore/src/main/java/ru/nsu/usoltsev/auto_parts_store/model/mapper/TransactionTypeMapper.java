package ru.nsu.usoltsev.auto_parts_store.model.mapper;

import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;
import ru.nsu.usoltsev.auto_parts_store.model.dto.TransactionTypeDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.TransactionType;

@Mapper
public interface TransactionTypeMapper {
    TransactionTypeMapper INSTANCE = Mappers.getMapper(TransactionTypeMapper.class);

    TransactionTypeDto toDto(TransactionType transactionType);

    TransactionType fromDto(TransactionTypeDto transactionTypeDto);
}
