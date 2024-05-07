import React, {useState} from "react";
import {Form, Input, Button, Table} from "antd";
import PostService from "../postService/PostService";

const SupplierQuery = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [supplierData, setSupplierData] = useState<{ suppliers: any[], count: number }>({suppliers: [], count: 0});

    const handleSubmit = (values: any) => {
        setActiveQuery(true);
        const {category} = values;
        // Получение данных
        PostService.getSuppliersByType(category).then((response: any) => {
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
                    rules={[{required: true, message: "Введите категорию"}]}
                >
                    <Input placeholder="Введите категорию"/>
                </Form.Item>

                <Form.Item>
                    <Button type="primary" htmlType="submit">
                        Получить список поставщиков
                    </Button>
                </Form.Item>
            </Form>
            <div style={{display: activeQuery ? "block" : "none"}}>
                <h2 style={{marginBottom: "15px"}}>Поставщики</h2>
                <p>Количество поставщиков: {supplierData.count}</p>
                <Table columns={columns} dataSource={supplierData.suppliers}/>
            </div>
        </div>
    );
};

export default SupplierQuery;
