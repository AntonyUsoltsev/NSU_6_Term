import React, {useState} from "react";
import {Form, Input, Button, Table} from "antd";
import PostService from "../../postService/PostService";

const SupplierByDeliveryQuery = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [supplierData, setSupplierData] = useState<[]>([]);

    const handleSubmit = (values: any) => {
        setActiveQuery(true);
        const {startDate, endDate, amount, itemName} = values;

        // Получение данных
        PostService.getSuppliersByDelivery(startDate, endDate, amount, itemName).then((response: any) => {
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
                    name="startDate"
                    rules={[{required: true, message: "Введите дату начала"}]}
                >
                    <Input placeholder="Введите дату начала в формате YYYY-MM-DD hh:mm:ss"/>
                </Form.Item>
                <Form.Item
                    name="endDate"
                    rules={[{required: true, message: "Введите дату конца"}]}
                >
                    <Input placeholder="Введите дату конца в формате YYYY-MM-DD hh:mm:ss"/>
                </Form.Item>
                <Form.Item
                    name="amount"
                    rules={[{required: true, message: "Введите количество"}]}
                >
                    <Input placeholder="Введите количество"/>
                </Form.Item>
                <Form.Item
                    name="itemName"
                    rules={[{required: true, message: "Введите название детали"}]}
                >
                    <Input placeholder="Введите название детали "/>
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

export default SupplierByDeliveryQuery;
