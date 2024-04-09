package ru.nsu.usoltsev.auto_parts_store.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.model.dto.CustomerDto;
import ru.nsu.usoltsev.auto_parts_store.model.dto.ItemsDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Customer;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Items;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.CustomerMapper;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.ItemsMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.ItemsRepository;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class ItemsService {
    @Autowired
    private ItemsRepository itemsRepository;

    public ItemsDto saveItem(ItemsDto itemsDto) {
        Items customer = ItemsMapper.INSTANCE.fromDto(itemsDto);
        Items savedItem = itemsRepository.saveAndFlush(customer);
        return ItemsMapper.INSTANCE.toDto(savedItem);
    }
    public ItemsDto getItemById(Long id) {
        return ItemsMapper.INSTANCE.toDto(itemsRepository.findById(id).orElseThrow());
    }

    public List<ItemsDto> getItems() {
        return itemsRepository.findAll()
                .stream()
                .map(ItemsMapper.INSTANCE::toDto)
                .collect(Collectors.toList());
    }
}