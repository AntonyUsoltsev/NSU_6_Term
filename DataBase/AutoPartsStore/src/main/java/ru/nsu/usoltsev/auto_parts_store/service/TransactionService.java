package ru.nsu.usoltsev.auto_parts_store.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.model.Params;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CashierDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CustomerDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.OrdersDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.TransactionDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.AverageSellDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.CashReportDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.SellingSpeedDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.TransactionInfoDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Cashier;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Orders;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Transaction;
import ru.nsu.usoltsev.auto_parts_store.model.entity.TransactionType;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.CashierMapper;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.OrdersMapper;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.TransactionMapper;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.TransactionTypeMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.CashierRepository;
import ru.nsu.usoltsev.auto_parts_store.repository.OrdersRepository;
import ru.nsu.usoltsev.auto_parts_store.repository.TransactionRepository;
import ru.nsu.usoltsev.auto_parts_store.repository.TransactionTypeRepository;

import java.sql.Timestamp;
import java.time.Duration;
import java.time.LocalDate;
import java.time.Period;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class TransactionService implements CrudService<TransactionDto> {

    @Autowired
    TransactionRepository transactionRepository;
    @Autowired
    OrdersRepository ordersRepository;
    @Autowired
    OrdersService ordersService;
    @Autowired
    CashierRepository cashierRepository;
    @Autowired
    TransactionTypeRepository transactionTypeRepository;

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
                    new CashierDto((Long) row[4], (String) row[5], (String) row[6]),
                    new CustomerDto((Long) row[7], (String) row[8], (String) row[9], (String) row[10]),
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

    @Override
    public List<TransactionDto> getAll() {
        List<TransactionDto> transactionDtos = new ArrayList<>();
        List<Transaction> transactions = transactionRepository.findAll();
        for (Transaction transaction : transactions) {
            TransactionDto transactionDto = TransactionMapper.INSTANCE.toDto(transaction);
            OrdersDto ordersDto = ordersService.findOrdersWithCustomer(transaction.getOrderId());
            Cashier cashier = cashierRepository.findById(transaction.getCashierId()).orElseThrow();
            TransactionType transactionType = transactionTypeRepository.findByTypeId(transaction.getTypeId());
            transactionDto.setOrders(ordersDto);
            transactionDto.setCashier(CashierMapper.INSTANCE.toDto(cashier));
            transactionDto.setTransactionTypeDto(TransactionTypeMapper.INSTANCE.toDto(transactionType));
            transactionDtos.add(transactionDto);
        }
        return transactionDtos;
    }

    @Override
    public void delete(Long id) {

    }

    @Override
    public TransactionDto add(TransactionDto dto) {
        return null;
    }

    @Override
    public void update(Long id, TransactionDto dto) {

    }
}
