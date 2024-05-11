import React, { useState } from "react";
import { Form, Button, Table } from "antd";
import PostService from "../postService/PostService";

const InventoryListQuery = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [itemsData, setItemData] = useState<[]>([]);

    const handleSubmit = () => {
        setActiveQuery(true);
        PostService.getInventoryList().then((response: any) => {
            setItemData(response.data);
        });
    };

    const columns = [
        {
            title: "Название детали",
            dataIndex: "name",
            key: "name",
        },
        {
            title: "Количество",
            dataIndex: "amount",
            key: "amount",
        },
        {
            title: "Категория",
            dataIndex: "categoryId",
            key: "categoryId",
        },
        {
            title: "Количество бракованных",
            dataIndex: "defectAmount",
            key: "defectAmount",
        },
        {
            title: "Цена",
            dataIndex: "price",
            key: "price",
        },
        {
            title: "Номер ячейки",
            dataIndex: "cellNumber",
            key: "cellNumber",
        },
    ];

    return (
        <div>
            <Form name="getInventory" onFinish={handleSubmit} form={form}>
                <Form.Item>
                    <Button type="primary" htmlType="submit">
                        Получить ведомость
                    </Button>
                </Form.Item>
            </Form>
            <div style={{ display: activeQuery ? "block" : "none" }}>
                <h2 style={{ marginBottom: "15px" }}>Детали</h2>
                <Table columns={columns} dataSource={itemsData} />
            </div>
        </div>
    );
};

export default InventoryListQuery;
