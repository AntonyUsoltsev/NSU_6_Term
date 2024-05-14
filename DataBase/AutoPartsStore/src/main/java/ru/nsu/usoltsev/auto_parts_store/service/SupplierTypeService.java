package ru.nsu.usoltsev.auto_parts_store.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierTypeDto;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.SupplierTypeMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.SupplierTypeRepository;

import java.util.List;

@Service
public class SupplierTypeService {
    @Autowired
    private SupplierTypeRepository supplierTypeRepository;

    public List<SupplierTypeDto> getSupplierTypes() {
        return supplierTypeRepository.findAll().stream()
                .map(SupplierTypeMapper.INSTANCE::toDto)
                .toList();
    }

    public void deleteSupplierType(Long id) {
        supplierTypeRepository.deleteById(id);
    }

    public void addSupplierType(SupplierTypeDto supplierTypeDto) {
        supplierTypeRepository.addSupplierType(supplierTypeDto.getTypeName());
    }

    public void updateSupplierType(Long id, SupplierTypeDto supplierTypeDto) {
        supplierTypeRepository.updateTypeNameById(id, supplierTypeDto.getTypeName());
    }
}
