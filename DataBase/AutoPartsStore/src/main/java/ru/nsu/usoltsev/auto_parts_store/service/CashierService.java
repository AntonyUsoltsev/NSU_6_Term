package ru.nsu.usoltsev.auto_parts_store.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.exception.ResourceNotFoundException;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CashierDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Cashier;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.CashierMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.CashierRepository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@Slf4j
public class CashierService implements CrudService<CashierDto> {

    @Autowired
    private CashierRepository cashierRepository;

    public CashierDto getCashierById(Long id) {
        return CashierMapper.INSTANCE.toDto(cashierRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Cashier is not found by id: " + id)));
    }

    @Override
    public List<CashierDto> getAll() {
        return cashierRepository.findAll()
                .stream()
                .map(CashierMapper.INSTANCE::toDto)
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {

    }

    @Override
    public CashierDto add(CashierDto dto) {
        Cashier cashier = CashierMapper.INSTANCE.fromDto(dto);
        Cashier savedCashier = cashierRepository.saveAndFlush(cashier);
        return CashierMapper.INSTANCE.toDto(savedCashier);
    }

    @Override
    public void update(Long id, CashierDto dto) {
        Optional<Cashier> optionalCashier = cashierRepository.findById(id);
        if (optionalCashier.isPresent()) {
            Cashier cashier = optionalCashier.get();
            cashier.setName(dto.getName());
            cashier.setSecondName(dto.getSecondName());
            cashierRepository.saveAndFlush(cashier);
        } else {
            throw new IllegalArgumentException("Customer with id=" + id + " not found");
        }
    }
}
