import React, {useState} from "react";
import {Form, Input, Button, Table} from "antd";
import PostService from "../../postService/PostService";

const RealisedItemsByDayQuery = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [itemsData, setItemData] = useState<[]>([]);

    const handleSubmit = (values: any) => {
        setActiveQuery(true);
        const {day} = values;
        // Получение данных
        PostService.getRealisedItems(day).then((response: any) => {
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
            title: "Цена",
            dataIndex: "price",
            key: "price",
        }
    ];

    return (
        <div>
            <Form name="getSuppliers" onFinish={handleSubmit} form={form}>
                <Form.Item
                    name="day"
                    rules={[{required: true, message: "Введите день "}]}
                >
                    <Input placeholder="Введите день в формате YYYY-MM-DD"/>
                </Form.Item>

                <Form.Item>
                    <Button type="primary" htmlType="submit">
                        Получить список деталей
                    </Button>
                </Form.Item>
            </Form>
            <div style={{display: activeQuery ? "block" : "none"}}>
                <h2 style={{marginBottom: "15px"}}>Детали</h2>
                <Table columns={columns} dataSource={itemsData}/>
            </div>
        </div>
    );
};

export default RealisedItemsByDayQuery;
