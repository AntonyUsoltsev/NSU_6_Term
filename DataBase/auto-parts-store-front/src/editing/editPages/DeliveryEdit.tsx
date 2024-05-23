import React, { useEffect, useState } from "react";
import { Form, Button, Table, Input, Popconfirm, message, Modal, Select, DatePicker } from "antd";
import moment from "moment";

import PostService from "../../postService/PostService";

const DeliveryEdit: React.FC = () => {
    const [deliveriesData, setDeliveriesData] = useState([]);
    const [suppliersData, setSuppliersData] = useState([]);
    const [itemsData, setItemsData] = useState([]);
    const [editMode, setEditMode] = useState(false);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [form] = Form.useForm();
    const [currentDelivery, setCurrentDelivery] = useState<any>();

    useEffect(() => {
        fetchDeliveries();
        fetchSuppliers();
        fetchItems();
    }, []);

    const fetchDeliveries = () => {
        PostService.getRequest(`delivery/all`).then((response: any) => {
            setDeliveriesData(response.data);
        });
    };

    const fetchSuppliers = () => {
        PostService.getRequest(`suppliers/all`).then((response: any) => {
            setSuppliersData(response.data);
        });
    };

    const fetchItems = () => {
        PostService.getRequest(`items/all`).then((response: any) => {
            setItemsData(response.data);
        });
    };

    const handleSave = async (values: any) => {
        try {
            const body = {
                supplierId: values.supplierId,
                deliveryDate: values.deliveryDate.format('YYYY-MM-DDTHH:mm:ss.SSSZ'),
                itemsDelivery: values.itemsDelivery.map((item: any) => ({
                    item: {
                        itemId: item.itemId.key ? item.itemId.key : item.itemId,
                    },
                    purchasePrice: item.purchasePrice,
                })),
            };

            console.log(body)
            if (editMode && currentDelivery) {
                await PostService.updateRequest(`delivery/${currentDelivery.deliveryId}`, body);
            } else {
                await PostService.addRequest(`delivery`, body);
            }

            fetchDeliveries();
            resetForm();
        } catch (error) {
            message.error("Failed to save the delivery.");
        }
    };

    const handleDelete = async (deliveryId: number) => {
        try {
            await PostService.deleteRequest(`delivery/${deliveryId}`);
            message.success("Deleted delivery.");
            fetchDeliveries();
        } catch (error) {
            message.error("Failed to delete the delivery.");
        }
    };

    const resetForm = () => {
        setEditMode(false);
        setIsModalVisible(false);
        form.resetFields();
        setCurrentDelivery(null);
    };

    const handleAdd = () => {
        setEditMode(false);
        setIsModalVisible(true);
    };

    const handleEdit = (record: any) => {
        setEditMode(true);
        setIsModalVisible(true);
        setCurrentDelivery(record);
        form.setFieldsValue({
            supplierId: record.supplierId,
            deliveryDate: moment(record.deliveryDate),
            itemsDelivery: record.itemsDelivery.map((item: any) => ({
                itemId: {
                    key: item.item.itemId,
                    label: `${item.item.name} (Ячейка: ${item.item.cellNumber})`,
                },
                purchasePrice: item.purchasePrice,
            })),
        });
    };

    const getAvailableItems = () => {
        const deliveredItemIds = new Set(
            deliveriesData.flatMap((delivery:any) => delivery.itemsDelivery.map((item: any) => item.item.itemId))
        );

        return itemsData.filter((item: any) => !deliveredItemIds.has(item.itemId));
    };

    const columns = [
        {
            title: "Поставщик",
            dataIndex: ["supplier", "name"],
            key: "supplier",
        },
        {
            title: "Дата доставки",
            dataIndex: "deliveryDate",
            key: "deliveryDate",
            render: (text: string) => moment(text).format('YYYY-MM-DD'),
        },
        {
            title: "Детали доставки",
            dataIndex: "itemsDelivery",
            key: "itemsDelivery",
            render: (itemsDelivery: any[]) => (
                <div>
                    {itemsDelivery.map(item => (
                        <div key={item.item.itemId}>
                            {item.item.name} (Ячейка: {item.item.cellNumber}, Цена покупки: {item.purchasePrice})
                        </div>
                    ))}
                </div>
            ),
        },
        {
            title: "Действия",
            key: "actions",
            render: (text: any, record: any) => (
                <span>
                    <a onClick={() => handleEdit(record)} style={{ marginRight: "10px" }}>Редактировать</a>
                    <Popconfirm
                        title="Вы уверены, что хотите удалить эту доставку?"
                        onConfirm={() => handleDelete(record.deliveryId)}
                        okText="Да"
                        cancelText="Нет"
                    >
                        <a>Удалить</a>
                    </Popconfirm>
                </span>
            ),
        },
    ];

    return (
        <div>
            <h2 style={{ marginBottom: "15px" }}>Доставки</h2>
            <Button type="primary" onClick={handleAdd} style={{ marginBottom: "15px" }}>
                Добавить
            </Button>
            <Table
                columns={columns}
                dataSource={deliveriesData}
                rowKey="deliveryId"
            />
            <Modal
                title={editMode ? "Редактировать доставку" : "Добавить доставку"}
                visible={isModalVisible}
                onCancel={resetForm}
                footer={[
                    <Button key="back" onClick={resetForm}>
                        Отмена
                    </Button>,
                    <Button key="submit" type="primary" onClick={() => form.submit()}>
                        {editMode ? "Сохранить" : "Добавить"}
                    </Button>,
                ]}
            >
                <Form form={form} layout="vertical" onFinish={handleSave}>
                    <Form.Item
                        label="Поставщик"
                        name="supplierId"
                        rules={[{ required: true, message: "Пожалуйста, выберите поставщика" }]}
                    >
                        <Select>
                            {suppliersData.map((supplier: any) => (
                                <Select.Option key={supplier.supplierId} value={supplier.supplierId}>
                                    {supplier.name}
                                </Select.Option>
                            ))}
                        </Select>
                    </Form.Item>
                    <Form.Item
                        label="Дата доставки"
                        name="deliveryDate"
                        rules={[{ required: true, message: "Пожалуйста, выберите дату доставки" }]}
                    >
                        <DatePicker showTime format="YYYY-MM-DDTHH:mm:ss.SSSZ" />
                    </Form.Item>
                    <Form.List name="itemsDelivery">
                        {(fields, { add, remove }) => (
                            <>
                                {fields.map((field, index) => (
                                    <div key={field.key} style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                                        <Form.Item
                                            {...field}
                                            label={`Деталь #${index + 1}`}
                                            name={[field.name, 'itemId']}
                                            rules={[{ required: true, message: 'Пожалуйста, выберите деталь' }]}
                                            style={{ flex: 1, marginRight: '8px' }}
                                        >
                                            <Select>
                                                {getAvailableItems().map((item: any) => (
                                                    <Select.Option key={item.itemId} value={item.itemId}>
                                                        {item.name} (Ячейка: {item.cellNumber})
                                                    </Select.Option>
                                                ))}
                                            </Select>
                                        </Form.Item>
                                        <Form.Item
                                            {...field}
                                            label="Цена покупки"
                                            name={[field.name, 'purchasePrice']}
                                            rules={[{ required: true, message: 'Пожалуйста, введите цену покупки' }]}
                                            style={{ flex: 1, marginRight: '8px' }}
                                        >
                                            <Input type="number" />
                                        </Form.Item>
                                        <Button type="link" onClick={() => remove(field.name)}>Удалить</Button>
                                    </div>
                                ))}
                                <Form.Item>
                                    <Button type="dashed" onClick={() => add()} block>
                                        Добавить деталь
                                    </Button>
                                </Form.Item>
                            </>
                        )}
                    </Form.List>
                </Form>
            </Modal>
        </div>
    );
};

export default DeliveryEdit;
