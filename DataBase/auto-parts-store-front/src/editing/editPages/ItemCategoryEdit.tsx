import React, {useEffect, useState} from "react";
import {Form, Button, Table, Input, Popconfirm, message, Modal} from "antd";
import PostService from "../../postService/PostService";

const ItemCategoryEdit: React.FC = () => {
    const [itemCategoryData, setItemCategoryData] = useState([]);
    const [editMode, setEditMode] = useState(false);
    const [categoryName, setCategoryName] = useState("");
    const [selectedItemCategoryId, setSelectedItemCategoryId] = useState(null);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [form] = Form.useForm();

    useEffect(() => {
        fetchItemCategory();
    }, []);

    const fetchItemCategory = () => {
        PostService.getRequest(`itemCategory/all`).then((response: any) => {
            setItemCategoryData(response.data);
        });
    };

    const handleSave = async () => {
        try {
            const body = {
                categoryName: categoryName,
            }
            if (editMode) {
                await PostService.updateRequest(`itemCategory/${selectedItemCategoryId}`, body)
            } else {
                await PostService.addRequest(`itemCategory`, body)
            }
            fetchItemCategory();
            resetForm();
        } catch (error) {
            message.error("Failed to update item category");
        }
    };

    const handleDelete = async (ItemCategoryId: number) => {
        try {
            await PostService.deleteRequest(`itemCategory/${ItemCategoryId}`).then((response: any) => {
                fetchItemCategory();
            });
        } catch (error) {
            message.error("Failed to delete the item category.");
        }
    };

    const resetForm = () => {
        setEditMode(false);
        setCategoryName("");
        setSelectedItemCategoryId(null);
        setIsModalVisible(false);
    };

    const handleAdd = () => {
        setEditMode(false);
        setCategoryName("");
        setSelectedItemCategoryId(null);
        setIsModalVisible(true);
    };

    const columns = [
        {
            title: "Тип транзакции",
            dataIndex: "categoryName",
            key: "categoryName",
        },
        {
            title: "Действия",
            key: "actions",
            render: (text: any, record: any) => (
                <span>
                    <a onClick={() => {
                        setEditMode(true);
                        setCategoryName(record.categoryName);
                        setSelectedItemCategoryId(record.categoryId);
                        setIsModalVisible(true);
                    }}>Редактировать</a>
                    <Popconfirm
                        title="Вы уверены, что хотите удалить этот тип транзакции?"
                        onConfirm={() => handleDelete(record.categoryId)}
                        okText="Да"
                        cancelText="Нет"
                    >
                        <a style={{marginLeft: 8}}>Удалить</a>
                    </Popconfirm>
                </span>
            ),
        },
    ];

    return (
        <div>
            <h2 style={{marginBottom: "15px"}}>Категории деталей</h2>
            <Button type="primary" onClick={handleAdd} style={{marginBottom: "15px"}}>
                Добавить
            </Button>
            <Table columns={columns} dataSource={itemCategoryData}/>
            <Modal
                title={editMode ? "Редактировать категорию детали" : "Добавить категорию детали"}
                visible={isModalVisible}
                onOk={handleSave}
                onCancel={() => setIsModalVisible(false)}
                okText={editMode ? "Сохранить" : "Добавить"}
                cancelText="Отмена"
            >
                <Form layout="vertical" form={form}>
                    <Form.Item
                        label="Категория детали"
                        rules={[{required: true, message: "Пожалуйста, введите категорию"}]}
                    >
                        <Input value={categoryName} onChange={(e) => setCategoryName(e.target.value)}/>
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default ItemCategoryEdit;
