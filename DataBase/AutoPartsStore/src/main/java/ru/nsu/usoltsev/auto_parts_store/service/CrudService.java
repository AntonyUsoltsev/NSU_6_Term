package ru.nsu.usoltsev.auto_parts_store.service;

import jakarta.transaction.Transactional;

import java.util.List;

public interface CrudService<D> {
    List<D> getAll();

    void delete(Long id);

    D add(D dto);

    void update(Long id, D dto);
}
