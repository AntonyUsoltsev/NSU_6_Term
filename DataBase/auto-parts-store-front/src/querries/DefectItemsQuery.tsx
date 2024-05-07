import React, { useState } from "react";
import { Form, Input, Button, Table } from "antd";
import PostService from "../postService/PostService";

const DefectItemsQuery = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [defectItems, setDefectItems] = useState([]);

    const handleSubmit = (values: any) => {
        setActiveQuery(true);
        const { startDate, endDate } = values;
        // Получение данных
        PostService.getDefectItems(startDate, endDate).then((response: any) => {
            setDefectItems(response.data);
        });
    };

    const columns = [
        {
            title: "Наименование товара",
            dataIndex: "itemName",
            key: "itemName",
        },
        {
            title: "Количество дефектных",
            dataIndex: "defectAmount",
            key: "defectAmount",
        },
        {
            title: "Дата поставки",
            dataIndex: "deliveryDate",
            key: "deliveryDate",
        },
        {
            title: "Поставщик",
            dataIndex: "supplierName",
            key: "supplierName",
        }
    ];

    return (
        <div>
            <Form name="getCashReport" onFinish={handleSubmit} form={form}>
                <Form.Item
                    name="startDate"
                    rules={[{ required: true, message: "Введите дату начала" }]}
                >
                    <Input placeholder="Введите дату начала" />
                </Form.Item>
                <Form.Item
                    name="endDate"
                    rules={[{ required: true, message: "Введите дату конца" }]}
                >
                    <Input placeholder="Введите дату конца" />
                </Form.Item>

                <Form.Item>
                    <Button type="primary" htmlType="submit">
                        Получить список бракованных деталей
                    </Button>
                </Form.Item>
            </Form>
            <div style={{ display: activeQuery ? "block" : "none" }}>
                <h2 style={{marginBottom: "15px"}}>Бракованные детали</h2>
                <Table columns={columns} dataSource={defectItems} />
            </div>
        </div>
    );
};

export default DefectItemsQuery;
