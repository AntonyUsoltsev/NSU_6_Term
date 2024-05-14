import React, {useState} from "react";
import {Form, Button, Table} from "antd";
import PostService from "../../postService/PostService";

const ItemDeliveryPriceInfo = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [itemsData, setItemsData] = useState<any[]>([]);

    const handleSubmit = () => {
        setActiveQuery(true);
        PostService.getItemsDeliveryPriceInfo().then((response: any) => {
            setItemsData(response.data);
        });
    };

    const columns = [
        {
            title: "Наименование",
            dataIndex: "name",
            key: "name",
        },
        {
            title: "Поставщик",
            dataIndex: "supplierDeliveryPriceList",
            key: "supplierDeliveryPriceList",
            render: (supplierList: any[]) => (
                <ul>
                    {supplierList.map((supplier, index) => (
                        <li key={index}>
                            {supplier.supplierName}:  {supplier.price} руб
                        </li>
                    ))}
                </ul>
            ),
        },
    ];

    return (
        <div>
            <Form name="getSuppliers" onFinish={handleSubmit} form={form}>
                <Form.Item>
                    <Button type="primary" htmlType="submit">
                        Получить список деталей
                    </Button>
                </Form.Item>
            </Form>
            <div style={{display: activeQuery ? "block" : "none"}}>
                <h2 style={{marginBottom: "15px"}}>Детали</h2>
                <Table columns={columns}
                       dataSource={itemsData}
                       pagination={{pageSize: 20}}/>
            </div>
        </div>
    );
};

export default ItemDeliveryPriceInfo;
