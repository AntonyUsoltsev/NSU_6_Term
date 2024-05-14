import React, {useState} from "react";
import {Form, Button, Table} from "antd";
import PostService from "../../postService/PostService";

const AverageSellQuery: React.FC  = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [itemsData, setItemData] = useState<[]>([]);

    const handleSubmit = () => {
        setActiveQuery(true);
        PostService.getAverageSell().then((response: any) => {
            setItemData(response.data);
        });
    };

    const columns = [
        {
            title: "Тип деталей",
            dataIndex: "typeName",
            key: "typeName",
        },
        {
            title: "Продано штук",
            dataIndex: "sellAmount",
            key: "sellAmount",
        },
        {
            title: "В среднем продано (шт/мес)",
            dataIndex: "averageSell",
            key: "averageSell",
        }
    ];

    return (
        <div>
            <Form name="getAverageSell" onFinish={handleSubmit} form={form}>
                <Form.Item>
                    <Button type="primary" htmlType="submit">
                        Получить средние продажи
                    </Button>
                </Form.Item>
            </Form>
            <div style={{display: activeQuery ? "block" : "none"}}>
                <h2 style={{marginBottom: "15px"}}>Продажи</h2>
                <Table columns={columns} dataSource={itemsData}/>
            </div>
        </div>
    );
};

export default AverageSellQuery;
