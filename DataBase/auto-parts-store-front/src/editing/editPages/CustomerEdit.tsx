import React, {useEffect, useState} from "react";
import {Form, Button, Table, Input, Popconfirm, message, Modal} from "antd";

import PostService from "../../postService/PostService";

const CustomerEdit: React.FC = () => {
    const [customersData, setCustomersData] = useState([]);
    const [editMode, setEditMode] = useState(false);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [form] = Form.useForm();
    const [currentCustomer, setCurrentCustomer] = useState<any>();

    useEffect(() => {
        fetchCustomers();
    }, []);

    const fetchCustomers = () => {
        PostService.getRequest(`customers/all`).then((response: any) => {
            setCustomersData(response.data);
        });
    };

    const handleSave = async (values: any) => {
        try {
            const body = {
                name: values.name,
                secondName: values.secondName,
                email: values.email,
            };

            if (editMode && currentCustomer) {
                await PostService.updateRequest(`customers/${currentCustomer.customerId}`, body);
            } else {
                await PostService.addRequest(`customers`, body);
            }
            fetchCustomers();
            resetForm();
        } catch (error) {
            message.error("Failed to save the customer.");
        }
    };

    const handleDelete = async (customerId: number) => {
        try {
            await PostService.deleteRequest(`customers/${customerId}`);
            message.success("Deleted customer.");
            fetchCustomers();
        } catch (error) {
            message.error("Failed to delete the customer.");
        }
    };

    const resetForm = () => {
        setEditMode(false);
        setIsModalVisible(false);
        form.resetFields();
        setCurrentCustomer(null);
    };

    const handleAdd = () => {
        setEditMode(false);
        setIsModalVisible(true);
    };

    const handleEdit = (record: any) => {
        setEditMode(true);
        setIsModalVisible(true);
        setCurrentCustomer(record);
        form.setFieldsValue({
            name: record.name,
            secondName: record.secondName,
            email: record.email,
        });
    };

    const columns = [
        {
            title: "Имя",
            dataIndex: "name",
            key: "name",
        },
        {
            title: "Фамилия",
            dataIndex: "secondName",
            key: "secondName",
        },
        {
            title: "Email",
            dataIndex: "email",
            key: "email",
        },
        {
            title: "Действия",
            key: "actions",
            render: (text: any, record: any) => (
                <span>
                    <a onClick={() => handleEdit(record)} style={{marginRight: "10px"}}>Редактировать</a>
                    <Popconfirm
                        title="Вы уверены, что хотите удалить этого клиента?"
                        onConfirm={() => handleDelete(record.customerId)}
                        okText="Да"
                        cancelText="Нет"
                    >
                        <a>Удалить</a>
                    </Popconfirm>
                </span>
            ),
        },
    ];

    return (
        <div>
            <h2 style={{marginBottom: "15px"}}>Покупатели</h2>
            <Button type="primary" onClick={handleAdd} style={{marginBottom: "15px"}}>
                Добавить
            </Button>
            <Table columns={columns} dataSource={customersData}/>
            <Modal
                title={editMode ? "Редактировать покупателя" : "Добавить покупателя"}
                visible={isModalVisible}
                onCancel={resetForm}
                footer={[
                    <Button key="back" onClick={resetForm}>
                        Отмена
                    </Button>,
                    <Button key="submit" type="primary" onClick={() => form.submit()}>
                        {editMode ? "Сохранить" : "Добавить"}
                    </Button>,
                ]}
            >
                <Form form={form} layout="vertical" onFinish={handleSave}>
                    <Form.Item
                        label="Имя"
                        name="name"
                        rules={[{required: true, message: "Пожалуйста, введите имя"}]}
                    >
                        <Input />
                    </Form.Item>
                    <Form.Item
                        label="Фамилия"
                        name="secondName"
                        rules={[{required: true, message: "Пожалуйста, введите фамилию"}]}
                    >
                        <Input />
                    </Form.Item>
                    <Form.Item
                        label="Email"
                        name="email"
                        rules={[{required: true, message: "Пожалуйста, введите email"}]}
                    >
                        <Input />
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default CustomerEdit;
