package ru.nsu.usoltsev.auto_parts_store.service;

import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.exception.ResourceNotFoundException;
import ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.SupplierByTypeDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Supplier;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.SupplierMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.SupplierRepository;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class SupplierService {

    @Autowired
    private SupplierRepository supplierRepository;


    public SupplierDto saveSupplier(@Valid SupplierDto supplierDto) {
        Supplier supplier = SupplierMapper.INSTANCE.fromDto(supplierDto);
        Supplier savedSupplier = supplierRepository.saveAndFlush(supplier);
        return SupplierMapper.INSTANCE.toDto(savedSupplier);
    }

    public SupplierDto getSupplierById(Long id) {
        return SupplierMapper.INSTANCE.toDto(supplierRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("supplier is not found by id: " + id)));
    }

    public List<SupplierDto> getSuppliers() {
        return supplierRepository.findAll()
                .stream()
                .map(SupplierMapper.INSTANCE::toDto)
                .collect(Collectors.toList());
    }

    public List<SupplierDto> getSuppliersByItemCategory(String category) {
        return supplierRepository.findSuppliersByItemCategory(category)
                .stream()
                .map(SupplierMapper.INSTANCE::toDto)
                .collect(Collectors.toList());
    }

    public SupplierByTypeDto getSuppliersByType(String type) {
        List<SupplierDto> supplierDtos =  supplierRepository.findSuppliersByType(type)
                .stream()
                .map(SupplierMapper.INSTANCE::toDto)
                .toList();
        Integer count = supplierRepository.findSuppliersCountByType(type);
        return new SupplierByTypeDto(supplierDtos, count);
    }

}
