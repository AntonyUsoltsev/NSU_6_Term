package ru.nsu.usoltsev.auto_parts_store.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.model.Params;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CashierDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CustomerDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.AverageSellDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.CashReportDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.SellingSpeedDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.TransactionInfoDto;
import ru.nsu.usoltsev.auto_parts_store.repository.TransactionRepository;

import java.sql.Timestamp;
import java.time.Duration;
import java.time.LocalDate;
import java.time.Period;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
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
                        (Long) row[2]))
                .toList();
    }

    public List<SellingSpeedDto> getSellingSpeed() {
        return transactionRepository.findSellingSpeed()
                .stream()
                .map(row -> new SellingSpeedDto(
                        (String) row[0],
                        (Timestamp) row[1],
                        (Timestamp) row[2],
                        ((Duration) row[3]).toDays()))
                .collect(Collectors.toList());
    }

    public List<CashReportDto> getCashReport(String fromDate, String toDate) {
        List<CashReportDto> report = new ArrayList<>();
        Timestamp fromTime = Timestamp.valueOf(fromDate);
        Timestamp toTime = Timestamp.valueOf(toDate);
        List<Object[]> transactionInfo = transactionRepository.findTransactionInfo(fromTime, toTime);
        for (Object[] row : transactionInfo) {
            List<CashReportDto.TransactionItemList> transactionOrderList = transactionRepository.findTransactionOrderList((Long) row[0])
                    .stream()
                    .map(array -> new CashReportDto.TransactionItemList(
                            (String) array[0],
                            (Long) array[1],
                            (String) array[2],
                            (Long) array[3]))
                    .toList();
            report.add(new CashReportDto(
                    (Timestamp) row[1],
                    (String) row[2],
                    (Integer) row[3],
                    new CashierDto((String) row[4], (String) row[5]),
                    new CustomerDto((String) row[6], (String) row[7], (String) row[8]),
                    transactionOrderList)
            );
        }
        return report;
    }

    public List<AverageSellDto> getAverageSell() {


        LocalDate currentDate = LocalDate.now();
        LocalDate foundingDate = Params.storeFoundingDate.toLocalDateTime().toLocalDate();

        Period period = Period.between(foundingDate, currentDate);
        int months = period.getYears() * 12 + period.getMonths();
        System.out.println(months);
        return transactionRepository.findAverageSell()
                .stream()
                .map(row -> new AverageSellDto(
                        (String) row[0],
                        (Long) row[1],
                        (Double.valueOf((Long) row[1]) / months)))
                .toList();


    }
}
