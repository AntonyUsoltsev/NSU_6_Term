package ru.nsu.usoltsev.auto_parts_store.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierTypeDto;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.SupplierTypeMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.SupplierTypeRepository;

import java.util.List;

@Service
public class SupplierTypeService implements CrudService<SupplierTypeDto> {
    @Autowired
    private SupplierTypeRepository supplierTypeRepository;

    @Override
    public List<SupplierTypeDto> getAll() {
        return supplierTypeRepository.findAll().stream()
                .map(SupplierTypeMapper.INSTANCE::toDto)
                .toList();
    }

    @Override
    public void delete(Long id) {
        supplierTypeRepository.deleteById(id);
    }

    @Override
    public void add(SupplierTypeDto supplierTypeDto) {
        supplierTypeRepository.addSupplierType(supplierTypeDto.getTypeName());
    }

    @Override
    public void update(Long id, SupplierTypeDto supplierTypeDto) {
        supplierTypeRepository.updateTypeNameById(id, supplierTypeDto.getTypeName());
    }
}
