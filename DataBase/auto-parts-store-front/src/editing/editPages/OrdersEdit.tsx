import React, {useEffect, useState} from "react";
import {Form, Button, Table, Input, Popconfirm, message, Modal, Select, DatePicker} from "antd";
import moment from "moment";

import PostService from "../../postService/PostService";

const OrdersEdit: React.FC = () => {
    const [ordersData, setOrdersData] = useState([]);
    const [customersData, setCustomersData] = useState([]);
    const [itemsData, setItemsData] = useState([]);
    const [itemQuantities, setItemQuantities] = useState(new Map());
    const [editMode, setEditMode] = useState(false);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [form] = Form.useForm();
    const [currentOrder, setCurrentOrder] = useState<any>();

    useEffect(() => {
        fetchOrders();
        fetchCustomers();
        fetchItems();
    }, []);

    const fetchOrders = () => {
        PostService.getRequest(`orders/all`).then((response: any) => {
            setOrdersData(response.data);
        });
    };

    const fetchCustomers = () => {
        PostService.getRequest(`customers/all`).then((response: any) => {
            setCustomersData(response.data);
        });
    };

    const fetchItems = () => {
        PostService.getRequest(`items/all`).then((response: any) => {
            const items = response.data;
            setItemsData(items);
            const quantities = new Map();
            items.forEach((item: any) => {
                quantities.set(item.itemId, item.amount);  // Assuming each item has a quantity field
            });
            setItemQuantities(quantities);
        });
    };

    const handleSave = async (values: any) => {
        try {
            const body = {
                customerId: values.customerId,
                orderDate: values.orderDate.format('YYYY-MM-DDTHH:mm:ss.SSSZ'),
                itemsOrders: values.itemsOrders.map((item: any) => ({
                    item: {
                        itemId: item.itemId.key ? item.itemId.key : item.itemId,
                    },
                    amount: item.amount,
                })),
            };

            if (editMode && currentOrder) {
                await PostService.updateRequest(`orders/${currentOrder.orderId}`, body);
            } else {
                await PostService.addRequest(`orders`, body);
            }

            fetchOrders();
            resetForm();
        } catch (error) {
            message.error("Failed to save the order.");
        }
    };

    const handleDelete = async (orderId: number) => {
        try {
            await PostService.deleteRequest(`orders/${orderId}`);
            message.success("Deleted order.");
            fetchOrders();
        } catch (error) {
            message.error("Failed to delete the order.");
        }
    };

    const resetForm = () => {
        setEditMode(false);
        setIsModalVisible(false);
        form.resetFields();
        setCurrentOrder(null);
    };

    const handleAdd = () => {
        setEditMode(false);
        setIsModalVisible(true);
    };

    const handleEdit = (record: any) => {
        setEditMode(true);
        setIsModalVisible(true);
        setCurrentOrder(record);
        form.setFieldsValue({
            customerId: record.customerId,
            orderDate: moment(record.orderDate),
            itemsOrders: record.itemsOrders.map((order: any) => ({
                itemId: {
                    key: order.item.itemId,
                    label: `${order.item.name} (Ячейка: ${order.item.cellNumber})`,
                },
                amount: order.amount
            })),
        });
    };

    const getAvailableItems = () => {
        const orderedItemIds = new Set(
            ordersData.flatMap((order: any) => order.itemsOrders.map((item: any) => item.item.itemId))
        );

        return itemsData.filter((item: any) => !orderedItemIds.has(item.itemId));
    };

    const columns = [
        {
            title: "Покупатель",
            dataIndex: ["customer", "name"],
            key: "customer",
            render: (text: string, record: any) => `${record.customer.name} ${record.customer.secondName}`,
        },
        {
            title: "Дата заказа",
            dataIndex: "orderDate",
            key: "orderDate",
            render: (text: string) => moment(text).format('YYYY-MM-DD'),
        },
        {
            title: "Детали заказа",
            dataIndex: "itemsOrders",
            key: "itemsOrders",
            render: (itemsOrders: any[]) => (
                <div>
                    {itemsOrders.map(item => (
                        <div key={item.item.itemId}>
                            {item.item.name} (Ячейка: {item.item.cellNumber}, Цена: {item.item.price} р.,
                            Количество: {item.amount})
                        </div>
                    ))}
                </div>
            ),
        },
        {
            title: "Полная стоимость",
            dataIndex: "fullPrice",
            key: "fullPrice",
        },
        {
            title: "Действия",
            key: "actions",
            render: (text: any, record: any) => (
                <span>
                    <a onClick={() => handleEdit(record)} style={{marginRight: "10px"}}>Редактировать</a>
                    <Popconfirm
                        title="Вы уверены, что хотите удалить этот заказ?"
                        onConfirm={() => handleDelete(record.orderId)}
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
            <h2 style={{marginBottom: "15px"}}>Заказы</h2>
            <Button type="primary" onClick={handleAdd} style={{marginBottom: "15px"}}>
                Добавить
            </Button>
            <Table
                columns={columns}
                dataSource={ordersData}
                rowKey="orderId"
            />
            <Modal
                title={editMode ? "Редактировать заказ" : "Добавить заказ"}
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
                        label="Покупатель"
                        name="customerId"
                        rules={[{required: true, message: "Пожалуйста, выберите покупателя"}]}
                    >
                        <Select>
                            {customersData.map((customer: any) => (
                                <Select.Option key={customer.customerId} value={customer.customerId}>
                                    {customer.name} {customer.secondName}
                                </Select.Option>
                            ))}
                        </Select>
                    </Form.Item>
                    <Form.Item
                        label="Дата заказа"
                        name="orderDate"
                        rules={[{required: true, message: "Пожалуйста, выберите дату заказа"}]}
                    >
                        <DatePicker showTime format="YYYY-MM-DDTHH:mm:ss.SSSZ"/>
                    </Form.Item>
                    <Form.List name="itemsOrders">
                        {(fields, {add, remove}) => (
                            <>
                                {fields.map((field, index) => (
                                    <div key={field.key}
                                         style={{display: 'flex', alignItems: 'center', marginBottom: '10px'}}>
                                        <Form.Item
                                            {...field}
                                            label={`Деталь #${index + 1}`}
                                            name={[field.name, 'itemId']}
                                            rules={[{required: true, message: 'Пожалуйста, выберите деталь'}]}
                                            style={{flex: 1, marginRight: '8px'}}
                                        >
                                            <Select>
                                                {itemsData.map((item: any) => (
                                                    <Select.Option key={item.itemId} value={item.itemId}>
                                                        {item.name} (Ячейка: {item.cellNumber})
                                                    </Select.Option>
                                                ))}
                                            </Select>
                                        </Form.Item>

                                        <Form.Item
                                            {...field}
                                            label="Количество"
                                            name={[field.name, 'amount']}
                                            rules={[
                                                {required: true, message: 'Пожалуйста, введите количество'},
                                                ({getFieldValue}) => ({
                                                    validator(_, value) {
                                                        const itemId = getFieldValue(['itemsOrders', field.name, 'itemId']);
                                                        const availableAmount = itemQuantities.get(itemId);
                                                        if (!value || (value > 0 && value <= availableAmount)) {
                                                            return Promise.resolve();
                                                        }
                                                        return Promise.reject(new Error(`Максимальное количество: ${availableAmount}`));
                                                    }
                                                })
                                            ]}
                                            style={{flex: 1, marginRight: '8px'}}
                                        >
                                            <Input type="number" min={1}/>
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

export default OrdersEdit;
