package ru.nsu.usoltsev.auto_parts_store.service;

import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;
import ru.nsu.usoltsev.auto_parts_store.exception.ResourceNotFoundException;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CustomerDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Customer;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.CustomerMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.CustomerRepository;

import java.sql.Timestamp;
import java.util.List;
import java.util.stream.Collectors;

@Service
@Slf4j
@Validated
public class CustomerService {

    @Autowired
    private CustomerRepository customerRepository;

    public CustomerDto saveCustomer(@Valid CustomerDto customerDto) {
        Customer customer = CustomerMapper.INSTANCE.fromDto(customerDto);
        Customer savedCustomer = customerRepository.saveAndFlush(customer);
        return CustomerMapper.INSTANCE.toDto(savedCustomer);
    }

    public CustomerDto getCustomerById(Long id) {
        return CustomerMapper.INSTANCE.toDto(customerRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Customer is not found by id: " + id)));
    }

    public List<CustomerDto> getCustomers() {
        return customerRepository.findAll()
                .stream()
                .map(CustomerMapper.INSTANCE::toDto)
                .collect(Collectors.toList());
    }


    public List<CustomerDto> getCustomerByItem(String fromDate, String toDate, Integer amount, String item) {
        Timestamp fromTime = Timestamp.valueOf(fromDate);
        Timestamp toTime = Timestamp.valueOf(toDate);
        return  customerRepository.findCustomerByItem(fromTime, toTime, amount, item)
                .stream()
                .map(CustomerMapper.INSTANCE::toDto)
                .toList();
    }
}
