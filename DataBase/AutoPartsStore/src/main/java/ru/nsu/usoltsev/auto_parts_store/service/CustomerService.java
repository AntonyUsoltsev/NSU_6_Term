package ru.nsu.usoltsev.auto_parts_store.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CustomerDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Customer;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.CustomerMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.CustomerRepository;

import java.util.List;
import java.util.stream.Collectors;

@Service
@Slf4j
public class CustomerService {

    @Autowired
    private CustomerRepository customerRepository;

    public CustomerDto saveCustomer(CustomerDto customerDto) {
        Customer customer = CustomerMapper.INSTANCE.fromDto(customerDto);
        Customer savedCustomer = customerRepository.saveAndFlush(customer);
        return CustomerMapper.INSTANCE.toDto(savedCustomer);
    }

    public CustomerDto getCustomer(Long id) {
        return CustomerMapper.INSTANCE.toDto(customerRepository.findById(id).orElseThrow());
    }

    public List<CustomerDto> getCustomers() {
        return customerRepository.findAll()
                .stream()
                .map(CustomerMapper.INSTANCE::toDto)
                .collect(Collectors.toList());
    }

}
