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
import java.util.stream.Collectors;

@Service
@Slf4j
public class CashierService {

    @Autowired
    private CashierRepository cashierRepository;

    public CashierDto saveCashier(CashierDto cashierDto) {
        Cashier cashier = CashierMapper.INSTANCE.fromDto(cashierDto);
        Cashier savedCashier = cashierRepository.saveAndFlush(cashier);
        return CashierMapper.INSTANCE.toDto(savedCashier);
    }

    public CashierDto getCashierById(Long id) {
        return CashierMapper.INSTANCE.toDto(cashierRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Cashier is not found by id: " + id)));
    }

    public List<CashierDto> getCashiers() {
        return cashierRepository.findAll()
                .stream()
                .map(CashierMapper.INSTANCE::toDto)
                .collect(Collectors.toList());
    }

}
