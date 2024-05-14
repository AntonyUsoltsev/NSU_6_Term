package ru.nsu.usoltsev.auto_parts_store.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.model.dto.ItemCategoryDto;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.ItemCategoryMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.ItemCategoryRepository;

import java.util.List;

@Service
public class ItemCategoryService implements CrudService<ItemCategoryDto> {

    @Autowired
    ItemCategoryRepository itemCategoryRepository;

    @Override
    public List<ItemCategoryDto> getAll() {
        return itemCategoryRepository.findAll().stream()
                .map(ItemCategoryMapper.INSTANCE::toDto)
                .toList();
    }

    @Override
    public void delete(Long id) {
        itemCategoryRepository.deleteById(id);
    }

    @Override
    public void add(ItemCategoryDto dto) {
        itemCategoryRepository.addItemCategory(dto.getCategoryName());

    }

    @Override
    public void update(Long id, ItemCategoryDto dto) {
        itemCategoryRepository.updateTypeNameById(id, dto.getCategoryName());

    }
}
