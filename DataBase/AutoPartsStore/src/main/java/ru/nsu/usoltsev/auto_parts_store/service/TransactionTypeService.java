package ru.nsu.usoltsev.auto_parts_store.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.model.dto.TransactionTypeDto;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.SupplierTypeMapper;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.TransactionTypeMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.TransactionTypeRepository;

import java.util.List;

@Service
public class TransactionTypeService implements CrudService<TransactionTypeDto> {
    @Autowired
    TransactionTypeRepository transactionTypeRepository;

    @Override
    public List<TransactionTypeDto> getAll() {
        return transactionTypeRepository.findAll().stream()
                .map(TransactionTypeMapper.INSTANCE::toDto)
                .toList();
    }

    @Override
    public void delete(Long id) {
        transactionTypeRepository.deleteById(id);
    }

    @Override
    public TransactionTypeDto add(TransactionTypeDto dto) {
        transactionTypeRepository.addTransactionType(dto.getTypeName());
        return TransactionTypeMapper.INSTANCE.toDto(transactionTypeRepository.findByTypeName(dto.getTypeName()));
    }

    @Override
    public void update(Long id, TransactionTypeDto dto) {
        transactionTypeRepository.updateTypeNameById(id, dto.getTypeName());

    }
}
