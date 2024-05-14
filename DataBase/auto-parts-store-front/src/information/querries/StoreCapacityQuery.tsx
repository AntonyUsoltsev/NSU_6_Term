import React, {useState} from "react";
import {Form, Button, Table} from "antd";
import PostService from "../../postService/PostService";

const StoreCapacityQuery = () => {
    const [form] = Form.useForm();
    const [activeQuery, setActiveQuery] = useState<boolean>(false);
    const [storeCapacity, setStoreCapacity] = useState<[]>([]);

    const handleSubmit = () => {
        setActiveQuery(true);
        PostService.getStoreCapacity().then((response: any) => {
            setStoreCapacity(response.data);
        });
    };


    return (
        <div>
            <Form name="getCapacity" onFinish={handleSubmit} form={form}>
                <Form.Item>
                    <Button type="primary" htmlType="submit">
                        Получить количество свободных ячеек
                    </Button>
                </Form.Item>
            </Form>
            <div style={{display: activeQuery ? "block" : "none"}}>
                <h2 style={{marginBottom: "15px"}}>На складе {storeCapacity} свободных ячеек</h2>
            </div>
        </div>
    );
};

export default StoreCapacityQuery;
