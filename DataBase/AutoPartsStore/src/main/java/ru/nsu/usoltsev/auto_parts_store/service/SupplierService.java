package ru.nsu.usoltsev.auto_parts_store.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.exception.ResourceNotFoundException;
import ru.nsu.usoltsev.auto_parts_store.model.dto.SupplierDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.SupplierByTypeDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Supplier;
import ru.nsu.usoltsev.auto_parts_store.model.entity.SupplierType;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.SupplierMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.SupplierRepository;
import ru.nsu.usoltsev.auto_parts_store.repository.SupplierTypeRepository;

import java.sql.Timestamp;
import java.util.List;
import java.util.Optional;

@Service
public class SupplierService implements CrudService<SupplierDto> {

    @Autowired
    private SupplierRepository supplierRepository;

    @Autowired
    private SupplierTypeRepository supplierTypeRepository;


    public SupplierDto getSupplierById(Long id) {
        return SupplierMapper.INSTANCE.toDto(supplierRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("supplier is not found by id: " + id)));
    }

    public List<SupplierDto> getSuppliersByItemCategory(String category) {
        return supplierRepository.findSuppliersByItemCategory(category)
                .stream()
                .map(row -> new SupplierDto(
                        (Long) row[0],
                        (String) row[1],
                        (String) row[2],
                        (String) row[3],
                        (Boolean) row[4]
                ))
                .toList();
    }

    public SupplierByTypeDto getSuppliersByType(Long type) {
        List<SupplierDto> supplierDtos = supplierRepository.findSuppliersByType(type);
        Integer count = supplierRepository.findSuppliersCountByType(type);
        return new SupplierByTypeDto(supplierDtos, count);
    }

    public List<SupplierDto> getSuppliersByDelivery(String fromDate, String toDate, Integer amount, String item) {
        Timestamp fromTime = Timestamp.valueOf(fromDate);
        Timestamp toTime = Timestamp.valueOf(toDate);
        return supplierRepository.findSuppliersByDelivery(fromTime, toTime, amount, item);
    }

    @Override
    public List<SupplierDto> getAll() {
        return supplierRepository.findAllSuplliers();
    }

    @Override
    public void delete(Long id) {
//        supplierRepository.deleteById(id);
    }

    @Override
    public SupplierDto add(SupplierDto dto) {
        Supplier supplier = SupplierMapper.INSTANCE.fromDto(dto);
        SupplierType supplierType = supplierTypeRepository.findByTypeName(dto.getTypeName());
        supplier.setTypeId(supplierType.getTypeId());
        Supplier savedSupplier = supplierRepository.saveAndFlush(supplier);
        return SupplierMapper.INSTANCE.toDto(savedSupplier);
    }

    @Override
    public void update(Long id, SupplierDto dto) {
        Optional<Supplier> optionalSupplier = supplierRepository.findById(id);
        SupplierType supplierType = supplierTypeRepository.findByTypeName(dto.getTypeName());
        if (optionalSupplier.isPresent()) {
            Supplier supplier = optionalSupplier.get();
            supplier.setName(dto.getName());
            supplier.setGaranty(dto.getGaranty());
            supplier.setTypeId(supplierType.getTypeId());
            supplier.setDocuments(dto.getDocuments());
            supplierRepository.saveAndFlush(supplier);
        } else {
            throw new IllegalArgumentException("Supplier with id=" + id + " not found");
        }
    }

}
