package ru.nsu.usoltsev.auto_parts_store.model.mapper;

import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CashierDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Cashier;

@Mapper
public interface CashierMapper {
    CashierMapper INSTANCE = Mappers.getMapper(CashierMapper.class);

    CashierDto toDto(Cashier cashier);

    Cashier fromDto(CashierDto cashierDto);
}
