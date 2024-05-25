import React, {useEffect, useState} from "react";
import {Form, Button, Table, Input, Popconfirm, message, Modal, Select, DatePicker} from "antd";
import moment from "moment";

import PostService from "../../postService/PostService";

const TransactionsEdit: React.FC = () => {
    const [transactionsData, setTransactionsData] = useState([]);
    const [ordersData, setOrdersData] = useState([]);
    const [cashiersData, setCashiersData] = useState([]);
    const [transactionTypesData, setTransactionTypesData] = useState([]);
    const [editMode, setEditMode] = useState(false);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [form] = Form.useForm();
    const [currentTransaction, setCurrentTransaction] = useState<any>();

    useEffect(() => {
        fetchTransactions();
        fetchOrders();
        fetchCashiers();
        fetchTransactionTypes();
    }, []);

    const fetchTransactions = () => {
        PostService.getRequest(`transactions/all`).then((response: any) => {
            setTransactionsData(response.data);
        });
    };

    const fetchOrders = () => {
        PostService.getRequest(`orders/all`).then((response: any) => {
            setOrdersData(response.data);
        });
    };

    const fetchCashiers = () => {
        PostService.getRequest(`cashiers/all`).then((response: any) => {
            setCashiersData(response.data);
        });
    };

    const fetchTransactionTypes = () => {
        PostService.getRequest(`transactionType/all`).then((response: any) => {
            setTransactionTypesData(response.data);
        });
    };

    const handleSave = async (values: any) => {
        try {
            const body = {
                transactionDate: values.transactionDate.format('YYYY-MM-DDTHH:mm:ss.SSSZ'),
                orders: {
                    orderId: values.orderId,
                },
                cashier: {
                    cashierId: values.cashierId,
                },
                transactionTypeDto: {
                    typeId: values.typeId,
                },
            };

            if (editMode && currentTransaction) {
                await PostService.updateRequest(`transaction/${currentTransaction.transactionId}`, body);
            } else {
                await PostService.addRequest(`transaction`, body);
            }

            fetchTransactions();
            resetForm();
        } catch (error) {
            message.error("Failed to save the transaction.");
        }
    };

    const handleDelete = async (transactionId: number) => {
        try {
            await PostService.deleteRequest(`transaction/${transactionId}`);
            message.success("Deleted transaction.");
            fetchTransactions();
        } catch (error) {
            message.error("Failed to delete the transaction.");
        }
    };

    const resetForm = () => {
        setEditMode(false);
        setIsModalVisible(false);
        form.resetFields();
        setCurrentTransaction(null);
    };

    const handleAdd = () => {
        setEditMode(false);
        setIsModalVisible(true);
    };

    const handleEdit = (record: any) => {
        setEditMode(true);
        setIsModalVisible(true);
        setCurrentTransaction(record);
        form.setFieldsValue({
            transactionDate: moment(record.transactionDate),
            orderId: record.orders.orderId,
            cashierId: record.cashier.cashierId,
            typeId: record.transactionTypeDto.typeId,
        });
    };

    const columns = [
        {
            title: "Дата транзакции",
            dataIndex: "transactionDate",
            key: "transactionDate",
            render: (text: string) => moment(text).format('YYYY-MM-DD'),
        },
        {
            title: "Заказ",
            dataIndex: "orders",
            key: "orders",
            render: (order: any) => (
                <div>
                    {order.customer.name} {order.customer.secondName} (Дата заказа: {moment(order.orderDate).format('YYYY-MM-DD')}, Полная стоимость: {order.fullPrice} р.)
                </div>
            ),
        },
        {
            title: "Кассир",
            dataIndex: ["cashier", "name"],
            key: "cashier",
            render: (text: string, record: any) => `${record.cashier.name} ${record.cashier.secondName}`,
        },
        {
            title: "Тип транзакции",
            dataIndex: ["transactionTypeDto", "typeName"],
            key: "transactionTypeDto",
        },
        {
            title: "Действия",
            key: "actions",
            render: (text: any, record: any) => (
                <span>
                    <a onClick={() => handleEdit(record)} style={{marginRight: "10px"}}>Редактировать</a>
                    <Popconfirm
                        title="Вы уверены, что хотите удалить эту транзакцию?"
                        onConfirm={() => handleDelete(record.transactionId)}
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
            <h2 style={{marginBottom: "15px"}}>Транзакции</h2>
            <Button type="primary" onClick={handleAdd} style={{marginBottom: "15px"}}>
                Добавить
            </Button>
            <Table
                columns={columns}
                dataSource={transactionsData}
                rowKey="transactionId"
            />
            <Modal
                title={editMode ? "Редактировать транзакцию" : "Добавить транзакцию"}
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
                        label="Дата транзакции"
                        name="transactionDate"
                        rules={[{required: true, message: "Пожалуйста, выберите дату транзакции"}]}
                    >
                        <DatePicker showTime format="YYYY-MM-DDTHH:mm:ss.SSSZ"/>
                    </Form.Item>
                    <Form.Item
                        label="Заказ"
                        name="orderId"
                        rules={[{required: true, message: "Пожалуйста, выберите заказ"}]}
                    >
                        <Select>
                            {ordersData.map((order: any) => (
                                <Select.Option key={order.orderId} value={order.orderId}>
                                    {order.customer.name} {order.customer.secondName} (Дата: {moment(order.orderDate).format('YYYY-MM-DD')})
                                </Select.Option>
                            ))}
                        </Select>
                    </Form.Item>
                    <Form.Item
                        label="Кассир"
                        name="cashierId"
                        rules={[{required: true, message: "Пожалуйста, выберите кассира"}]}
                    >
                        <Select>
                            {cashiersData.map((cashier: any) => (
                                <Select.Option key={cashier.cashierId} value={cashier.cashierId}>
                                    {cashier.name} {cashier.secondName}
                                </Select.Option>
                            ))}
                        </Select>
                    </Form.Item>
                    <Form.Item
                        label="Тип транзакции"
                        name="typeId"
                        rules={[{required: true, message: "Пожалуйста, выберите тип транзакции"}]}
                    >
                        <Select>
                            {transactionTypesData.map((type: any) => (
                                <Select.Option key={type.typeId} value={type.typeId}>
                                    {type.typeName}
                                </Select.Option>
                            ))}
                        </Select>
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default TransactionsEdit;
