package ru.nsu.usoltsev.auto_parts_store.controllers;

import lombok.AllArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CustomerDto;
import ru.nsu.usoltsev.auto_parts_store.service.CustomerService;

import java.util.List;

@RestController
@RequestMapping("api/customers")
@AllArgsConstructor
public class CustomerController {

    private CustomerService customerService;

    @GetMapping("/{id}")
    public ResponseEntity<CustomerDto> getCustomer(@PathVariable String id) {
        return ResponseEntity.ok(customerService.getCustomerById(Long.valueOf(id)));
    }

    @GetMapping()
    public ResponseEntity<List<CustomerDto>> getCustomers() {
        return ResponseEntity.ok(customerService.getCustomers());
    }

    @GetMapping("/byItemWithAmount")
    public ResponseEntity<List<CustomerDto>> getCustomerByItem(@RequestParam("from") String fromDate,
                                                               @RequestParam("to") String toDate,
                                                               @RequestParam("amount") Integer amount,
                                                               @RequestParam("item") String item) {
        return ResponseEntity.ok(customerService.getCustomerByItem(fromDate, toDate, amount, item));
    }
    @GetMapping("/byItem")
    public ResponseEntity<List<CustomerDto>> getCustomerByItem(@RequestParam("from") String fromDate,
                                                               @RequestParam("to") String toDate,
                                                               @RequestParam("item") String item) {
        return ResponseEntity.ok(customerService.getCustomerByItem(fromDate, toDate, 0, item));
    }


    @PostMapping()
    public ResponseEntity<CustomerDto> createCustomer(@RequestBody CustomerDto customerDto) {
        return new ResponseEntity<>(customerService.saveCustomer(customerDto), HttpStatus.CREATED);
    }
}
