package ru.nsu.usoltsev.auto_parts_store.service;

import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.nsu.usoltsev.auto_parts_store.model.dto.DeliveryDto;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Delivery;
import ru.nsu.usoltsev.auto_parts_store.model.entity.DeliveryList;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Item;
import ru.nsu.usoltsev.auto_parts_store.model.entity.Supplier;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.DeliveryMapper;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.ItemMapper;
import ru.nsu.usoltsev.auto_parts_store.model.mapper.SupplierMapper;
import ru.nsu.usoltsev.auto_parts_store.repository.DeliveryListRepository;
import ru.nsu.usoltsev.auto_parts_store.repository.DeliveryRepository;
import ru.nsu.usoltsev.auto_parts_store.repository.ItemRepository;
import ru.nsu.usoltsev.auto_parts_store.repository.SupplierRepository;

import java.util.ArrayList;
import java.util.List;

@Service
@Transactional
public class DeliveryService implements CrudService<DeliveryDto> {

    @Autowired
    DeliveryRepository deliveryRepository;

    @Autowired
    DeliveryListRepository deliveryListRepository;

    @Autowired
    ItemRepository itemRepository;

    @Autowired
    SupplierRepository supplierRepository;

    @Override
    public List<DeliveryDto> getAll() {
        List<DeliveryDto> deliveryDtos = new ArrayList<>();
        List<Delivery> deliveries = deliveryRepository.findAll();
        for (Delivery delivery : deliveries) {
            Supplier supplier = supplierRepository.findById(delivery.getSupplierId()).orElseThrow();
            List<DeliveryList> deliveryLists = deliveryListRepository.findDeliveriesByDeliveryId(delivery.getDeliveryId());
            List<DeliveryDto.ItemDeliveryDto> items = new ArrayList<>();
            for (DeliveryList deliveryList : deliveryLists) {
                items.add(new DeliveryDto.ItemDeliveryDto(
                        ItemMapper.INSTANCE.toDto(itemRepository.findById(deliveryList.getItemId()).orElseThrow()),
                        deliveryList.getPurchasePrice()));
            }
            DeliveryDto deliveryDto = DeliveryMapper.INSTANCE.toDto(delivery);
            deliveryDto.setSupplier(SupplierMapper.INSTANCE.toDto(supplier));
            deliveryDto.setItemsDelivery(items);
            deliveryDtos.add(deliveryDto);
        }
        return deliveryDtos;
    }

    @Override
    public void delete(Long id) {

    }

    @Override
    public DeliveryDto add(DeliveryDto dto) {
        Delivery delivery = new Delivery(dto.getDeliveryId(), dto.getSupplierId(), dto.getDeliveryDate());
        Delivery savedDelivery = deliveryRepository.saveAndFlush(delivery);
        for (DeliveryDto.ItemDeliveryDto itemDeliveryDto : dto.getItemsDelivery()) {
            Item item = itemRepository.findById(itemDeliveryDto.getItem().getItemId()).orElseThrow();
            DeliveryList deliveryList = new DeliveryList(
                    savedDelivery.getDeliveryId(),
                    itemDeliveryDto.getItem().getItemId(),
                    Long.valueOf(item.getAmount()),
                    itemDeliveryDto.getPurchasePrice()
            );
            deliveryListRepository.saveAndFlush(deliveryList);
        }
        return DeliveryMapper.INSTANCE.toDto(savedDelivery);
    }

    @Override
    public void update(Long id, DeliveryDto dto) {
        Delivery delivery = deliveryRepository.findById(id).orElseThrow();
        delivery.setDeliveryDate(dto.getDeliveryDate());
        delivery.setSupplierId(dto.getSupplierId());
        deliveryRepository.saveAndFlush(delivery);

        List<DeliveryList> existingDeliveryLists = deliveryListRepository.findDeliveriesByDeliveryId(id);
        deliveryListRepository.deleteAll(existingDeliveryLists);
        deliveryListRepository.flush();

        for (DeliveryDto.ItemDeliveryDto itemDeliveryDto : dto.getItemsDelivery()) {
            Item item = itemRepository.findById(itemDeliveryDto.getItem().getItemId()).orElseThrow();
            DeliveryList deliveryList = new DeliveryList(
                    id,
                    itemDeliveryDto.getItem().getItemId(),
                    Long.valueOf(item.getAmount()),
                    itemDeliveryDto.getPurchasePrice()
            );
            deliveryListRepository.saveAndFlush(deliveryList);
        }

    }
}
