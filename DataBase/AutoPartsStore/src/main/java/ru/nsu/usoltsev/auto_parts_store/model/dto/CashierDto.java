package ru.nsu.usoltsev.auto_parts_store.model.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.AllArgsConstructor;
import lombok.Data;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Transaction;

import java.util.List;

@Data
@AllArgsConstructor
public class CashierDto {

    private Long cashierId;

    @NotBlank(message = "Name cannot be blank")
    @Pattern(regexp = "[a-zA-Zа-яА-Я]+", message = "Name can contain only letters")
    private String name;

    @NotBlank(message = "Second name cannot be blank")
    @Pattern(regexp = "[a-zA-Zа-яА-Я]+", message = "Second name can contain only letters")
    private String secondName;

    private List<Transaction> transactions;
}