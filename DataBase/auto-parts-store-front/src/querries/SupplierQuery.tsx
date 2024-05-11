import React, {useState, useEffect} from "react";
import {Form, Button, Table, Select} from "antd";
import PostService from "../postService/PostService";

const {Option} = Select;

const SupplierQuery = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [supplierData, setSupplierData] = useState<{ suppliers: any[], count: number }>({suppliers: [], count: 0});
    const [categories, setCategories] = useState<any[]>([]);

    useEffect(() => {
        // Получение списка категорий при монтировании компонента
        PostService.getCategories().then((response: any) => {
            setCategories(response.data);
        });
    }, []);

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
                    rules={[{required: true, message: "Выберите категорию"}]}
                >
                    <Select placeholder="Выберите категорию">
                        {categories.map((category, index) => (
                            <Option key={index} value={category.typeId}>{category.typeName}</Option>
                        ))}
                    </Select>
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
