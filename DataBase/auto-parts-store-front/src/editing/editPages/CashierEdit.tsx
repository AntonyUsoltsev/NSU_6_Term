import React, {useEffect, useState} from "react";
import {Form, Button, Table, Input, Popconfirm, message, Modal} from "antd";

import PostService from "../../postService/PostService";

const CashierEdit: React.FC = () => {
    const [cashiersData, setCashiersData] = useState([]);
    const [editMode, setEditMode] = useState(false);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [form] = Form.useForm();
    const [currentCashier, setCurrentCashier] = useState<any>();

    useEffect(() => {
        fetchCashiers();
    }, []);

    const fetchCashiers = () => {
        PostService.getRequest(`cashiers/all`).then((response: any) => {
            setCashiersData(response.data);
        });
    };

    const handleSave = async (values: any) => {
        try {
            const body = {
                name: values.name,
                secondName: values.secondName,
            };

            if (editMode && currentCashier) {
                await PostService.updateRequest(`cashiers/${currentCashier.cashierId}`, body);
            } else {
                await PostService.addRequest(`cashiers`, body);
            }
            fetchCashiers();
            resetForm();
        } catch (error) {
            message.error("Failed to save the cashier.");
        }
    };

    const handleDelete = async (cashierId: number) => {
        try {
            await PostService.deleteRequest(`cashiers/${cashierId}`);
            message.success("Deleted cashier.");
            fetchCashiers();
        } catch (error) {
            message.error("Failed to delete the cashier.");
        }
    };

    const resetForm = () => {
        setEditMode(false);
        setIsModalVisible(false);
        form.resetFields();
        setCurrentCashier(null);
    };

    const handleAdd = () => {
        setEditMode(false);
        setIsModalVisible(true);
    };

    const handleEdit = (record: any) => {
        setEditMode(true);
        setIsModalVisible(true);
        setCurrentCashier(record);
        form.setFieldsValue({
            name: record.name,
            secondName: record.secondName,
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
            title: "Действия",
            key: "actions",
            render: (text: any, record: any) => (
                <span>
                    <a onClick={() => handleEdit(record)} style={{marginRight: "10px"}}>Редактировать</a>
                    <Popconfirm
                        title="Вы уверены, что хотите удалить этого кассира?"
                        onConfirm={() => handleDelete(record.cashierId)}
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
            <h2 style={{marginBottom: "15px"}}>Кассиры</h2>
            <Button type="primary" onClick={handleAdd} style={{marginBottom: "15px"}}>
                Добавить
            </Button>
            <Table columns={columns} dataSource={cashiersData}/>
            <Modal
                title={editMode ? "Редактировать кассира" : "Добавить кассира"}
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
                </Form>
            </Modal>
        </div>
    );
};

export default CashierEdit;
