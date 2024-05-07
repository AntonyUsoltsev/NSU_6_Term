import React, {useState} from "react";
import {Form, Input, Button, Table} from "antd";
import PostService from "../postService/PostService";

const SupplierByItemQuery = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [supplierData, setSupplierData] = useState<[]>([]);

    const handleSubmit = (values: any) => {
        setActiveQuery(true);
        const {category} = values;
        // Получение данных
        PostService.getSuppliersByItemType(category).then((response: any) => {
            setSupplierData(response.data);
        });
    };

    const columns = [
        {
            title: "Наименование поставщика",
            dataIndex: "name",
            key: "name",
        },
        {
            title: "Документы",
            dataIndex: "documents",
            key: "documents",
        },
        {
            title: "Тип",
            dataIndex: "typeName",
            key: "typeName",
        },
        {
            title: "Гарантия",
            dataIndex: "garanty",
            key: "garanty",
            render: (garanty: boolean) => (garanty ? "Есть" : "Нет"),
        }
    ];

    return (
        <div>
            <Form name="getSuppliers" onFinish={handleSubmit} form={form}>
                <Form.Item
                    name="category"
                    rules={[{required: true, message: "Введите категорию товара "}]}
                >
                    <Input placeholder="Введите категорию товара"/>
                </Form.Item>

                <Form.Item>
                    <Button type="primary" htmlType="submit">
                        Получить список поставщиков
                    </Button>
                </Form.Item>
            </Form>
            <div style={{display: activeQuery ? "block" : "none"}}>
                <h2 style={{marginBottom: "15px"}}>Поставщики</h2>
                <Table columns={columns} dataSource={supplierData}/>
            </div>
        </div>
    );
};

export default SupplierByItemQuery;
