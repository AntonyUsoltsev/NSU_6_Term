import React, { useState } from "react";
import { Form, Input, Button, Table } from "antd";
import PostService from "../../postService/PostService";

const CashReportQuery = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [cashReport, setCashReport] = useState([]);

    const handleSubmit = (values: any) => {
        setActiveQuery(true);
        const { startDate, endDate } = values;
        // Получение данных
        PostService.getCashReport(startDate, endDate).then((response: any) => {
            setCashReport(response.data);
        });
    };

    const columns = [
        {
            title: "",
            dataIndex: "transactionDate",
            key: "transactionDate",
        },
        {
            title: "Тип транзакции",
            dataIndex: "transactionType",
            key: "transactionType",
        },
        {
            title: "Полная цена",
            dataIndex: "fullPrice",
            key: "fullPrice",
            sorter: (a: any, b: any) => a.fullPrice - b.fullPrice,
        },
        {
            title: "Кассир",
            dataIndex: "cashier",
            key: "cashier",
            render: (cashier: any) =>
                `${cashier.name} ${cashier.secondName}`,
        },
        {
            title: "Клиент",
            dataIndex: "customer",
            key: "customer",
            render: (customer: any) =>
                `${customer.name} ${customer.secondName}`,
        },
        {
            title: "Товары",
            dataIndex: "itemList",
            key: "itemList",
            render: (itemList: any[]) => (
                <ul>
                    {itemList.map((item: any, itemIndex: number) => (
                        <li key={itemIndex}>
                            {`${item.itemName} (${item.amount}x ${item.price})`}
                        </li>
                    ))}
                </ul>
            ),
        },
    ];

    return (
        <div>
            <Form name="getCashReport" onFinish={handleSubmit} form={form}>
                <Form.Item
                    name="startDate"
                    rules={[{ required: true, message: "Введите дату начала" }]}
                >
                    <Input placeholder="Введите дату начала в формате YYYY-MM-DD hh:mm:ss" />
                </Form.Item>
                <Form.Item
                    name="endDate"
                    rules={[{ required: true, message: "Введите дату конца " }]}
                >
                    <Input placeholder="Введите дату конца в формате YYYY-MM-DD hh:mm:ss" />
                </Form.Item>

                <Form.Item>
                    <Button type="primary" htmlType="submit">
                        Получить кассовый отчет
                    </Button>
                </Form.Item>
            </Form>
            <div style={{ display: activeQuery ? "block" : "none" }}>
                <h2 style={{marginBottom: "15px"}}>Кассовый отчет</h2>
                <Table columns={columns} dataSource={cashReport} />
            </div>
        </div>
    );
};

export default CashReportQuery;
