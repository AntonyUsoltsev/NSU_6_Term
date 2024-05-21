import React, {useEffect, useState} from "react";
import {Form, Button, Table, Input, Popconfirm, message, Modal} from "antd";
import PostService from "../../postService/PostService";

const TransactionTypeEdit: React.FC = () => {
    const [transactionTypesData, setTransactionTypesData] = useState([]);
    const [editMode, setEditMode] = useState(false);
    const [typeName, setTypeName] = useState("");
    const [selectedTransactionTypeId, setSelectedTransactionTypeId] = useState(null);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [form] = Form.useForm();

    useEffect(() => {
        fetchTransactionTypes();
    }, []);

    const fetchTransactionTypes = () => {
        PostService.getRequest(`transactionType/all`).then((response: any) => {
            setTransactionTypesData(response.data);
        });
    };

    const handleSave = async () => {
        try {
            const body = {
                typeName: typeName,
            }
            if (editMode) {
                await PostService.updateRequest(`transactionType/${selectedTransactionTypeId}`, body);
            } else {
                await PostService.addRequest(`transactionType`, body);
            }
            fetchTransactionTypes();
            resetForm();
        } catch (error) {
            message.error("Failed to save the transaction type.");
        }
    };

    const handleDelete = async (TransactionTypeId: number) => {
        try {
            await PostService.deleteRequest(`transactionType/${TransactionTypeId}`).then((response: any) => {
                fetchTransactionTypes();
            });
        } catch (error) {
            message.error("Failed to delete the transaction type.");
        }
    };

    const resetForm = () => {
        setEditMode(false);
        setTypeName("");
        setSelectedTransactionTypeId(null);
        setIsModalVisible(false);
    };

    const columns = [
        {
            title: "Тип транзакции",
            dataIndex: "typeName",
            key: "typeName",
        },
        {
            title: "Действия",
            key: "actions",
            render: (text: any, record: any) => (
                <span>
                    <a onClick={() => {
                        setEditMode(true);
                        setTypeName(record.typeName);
                        setSelectedTransactionTypeId(record.typeId);
                        setIsModalVisible(true);
                    }}>Редактировать</a>
                    <Popconfirm
                        title="Вы уверены, что хотите удалить этот тип транзакции?"
                        onConfirm={() => handleDelete(record.typeId)}
                        okText="Да"
                        cancelText="Нет"
                    >
                        <a style={{marginLeft: 8}}>Удалить</a>
                    </Popconfirm>
                </span>
            ),
        },
    ];

    const handleAdd = () => {
        setEditMode(false);
        setTypeName("");
        setSelectedTransactionTypeId(null);
        setIsModalVisible(true);
    };

    return (
        <div>
            <h2 style={{marginBottom: "15px"}}>Категории транзакций</h2>
            <Button type="primary" onClick={handleAdd} style={{marginBottom: "15px"}}>
                Добавить
            </Button>
            <Table columns={columns} dataSource={transactionTypesData}/>
            <Modal
                title={editMode ? "Редактировать тип транзакции" : "Добавить тип транзакции"}
                visible={isModalVisible}
                onOk={handleSave}
                onCancel={() => setIsModalVisible(false)}
                okText={editMode ? "Сохранить" : "Добавить"}
                cancelText="Отмена"
            >
                <Form form={form} layout="vertical">
                    <Form.Item
                        label="Тип транзакции"
                        rules={[{required: true, message: "Пожалуйста, введите тип"}]}
                    >
                        <Input value={typeName} onChange={(e) => setTypeName(e.target.value)}/>
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default TransactionTypeEdit;
