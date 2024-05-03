package ru.nsu.usoltsev.auto_parts_store.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.TransactionInfoDto;
import ru.nsu.usoltsev.auto_parts_store.repository.TransactionRepository;

import java.sql.Timestamp;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class TransactionService {

    @Autowired
    TransactionRepository transactionRepository;

    public List<TransactionInfoDto> getTransactionInfoByDay(String date) {
        LocalDate localDate = LocalDate.parse(date);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy/MM/dd");
        String formattedDate = localDate.format(formatter);
        System.out.println(formattedDate);
        List<Object[]> resultList = transactionRepository.findRealiseItemsByDay(formattedDate);
        System.out.println(resultList);
        return resultList.stream()
                .map(row -> new TransactionInfoDto(
                        (String) row[0],
                        (Long) row[1],
                        (Long) row[2]
                )).toList();
    }

}
