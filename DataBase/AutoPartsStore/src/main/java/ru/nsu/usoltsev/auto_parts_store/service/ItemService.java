package ru.nsu.usoltsev.auto_parts_store.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.exception.ResourceNotFoundException;
import ru.nsu.usoltsev.auto_parts_store.model.dto.ItemDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.querriesDto.*;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Item;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.ItemMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.ItemRepository;

import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class ItemService {
    @Autowired
    private ItemRepository itemRepository;

    public ItemDto saveItem(ItemDto itemDto) {
        Item customer = ItemMapper.INSTANCE.fromDto(itemDto);
        Item savedItem = itemRepository.saveAndFlush(customer);
        return ItemMapper.INSTANCE.toDto(savedItem);
    }

    public ItemDto getItemById(Long id) {
        return ItemMapper.INSTANCE.toDto(itemRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Item is not found by id: " + id)));
    }

    public List<ItemDto> getItems() {
        return itemRepository.findAll()
                .stream()
                .map(ItemMapper.INSTANCE::toDto)
                .collect(Collectors.toList());
    }

    public List<ItemInfoDto> getItemsInfo() {
        return itemRepository.findAllItemsInfo()
                .stream()
                .map(row -> new ItemInfoDto(
                        (String) row[0],
                        (Integer) row[1],
                        (Integer) row[2]))
                .collect(Collectors.toList());
    }

    public List<ItemDto> getItemsByCategory(String category) {
        return itemRepository.findByCategory(category)
                .stream()
                .map(ItemMapper.INSTANCE::toDto)
                .collect(Collectors.toList());
    }

    public List<TopTenItemsDto> getTopTen() {
        return itemRepository.getTopTenSoldDetails()
                .stream()
                .map(row -> new TopTenItemsDto(
                        (String) row[0],
                        (Long) row[1]
                ))
                .toList();
    }


    public List<ItemDeliveryPriceDto> getItemDeliveryPrice() {
        List<ItemDeliveryPriceDto> itemDeliveryPriceList = new ArrayList<>();
        List<String> itemNames = itemRepository.getItemsNames();
        for (String name : itemNames) {
            List<ItemDeliveryPriceDto.SupplierDeliveryPrice> info = itemRepository.getSupplierPriceDateForItem(name)
                    .stream()
                    .map(row -> new ItemDeliveryPriceDto.SupplierDeliveryPrice(
                            (String) row[0],
                            (Integer) row[1],
                            (Timestamp) row[2]))
                    .toList();
            itemDeliveryPriceList.add(new ItemDeliveryPriceDto(name, info));
        }
        return itemDeliveryPriceList;
    }

    public List<DefectItemsDto> getDefectItems(String fromDate, String toDate) {
        Timestamp fromTime = Timestamp.valueOf(fromDate);
        Timestamp toTime = Timestamp.valueOf(toDate);
        return itemRepository.findDefectItems(fromTime, toTime)
                .stream()
                .map(row -> new DefectItemsDto(
                        (String) row[0],
                        (Integer) row[1],
                        (Timestamp) row[2],
                        (String) row[3]))
                .toList();
    }

    public List<ItemCatalogDto> getItemsCatalog() {
        List<ItemCatalogDto> ItemCatalogDtos = new ArrayList<>();
        List<Object[]> itemsInfo = itemRepository.getItemsCatalog();

        for (Object[] row : itemsInfo) {
            List<ItemCatalogDto.SupplierItemInfo> info = itemRepository.getSupplierItemInfo((String) row[0])
                    .stream()
                    .map(array -> new ItemCatalogDto.SupplierItemInfo(
                            (Integer) array[0],
                            (Integer) array[1],
                            (String) array[2]))
                    .toList();
            ItemCatalogDtos.add(new ItemCatalogDto(
                    (String) row[0],
                    (String) row[1],
                    info));
        }
        return ItemCatalogDtos;
    }

    public Integer getStoreCapacity() {
        return itemRepository.findStoreCapacity();
    }
}
