import React, {useState} from "react";
import {Form, Input, Button, Table} from "antd";
import PostService from "../postService/PostService";

const SellingSpeedQuery = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [sellData, setSellData] = useState<[]>([]);

    const handleSubmit = () => {
        setActiveQuery(true);
        PostService.getSellingSpeed().then((response: any) => {
            setSellData(response.data);
        });
    };

    const columns = [
        {
            title: "Название детали",
            dataIndex: "name",
            key: "name",
        },
        {
            title: "Дата продажи",
            dataIndex: "transactionDate",
            key: "transactionDate",
        },
        {
            title: "Дата поставки на склад",
            dataIndex: "deliveryDate",
            key: "deliveryDate",
        },
        {
            title: "Разница во времени (в днях)",
            dataIndex: "timeDiff",
            key: "timeDiff",
        }
    ];

    return (
        <div>
            <Form name="getSellingSpeed" onFinish={handleSubmit} form={form}>
                <Form.Item>
                    <Button type="primary" htmlType="submit">
                        Получить список товара
                    </Button>
                </Form.Item>
            </Form>
            <div style={{display: activeQuery ? "block" : "none"}}>
                <h2 style={{marginBottom: "15px"}}>Детали</h2>
                <Table columns={columns} dataSource={sellData}/>
            </div>
        </div>
    );
};

export default SellingSpeedQuery;
