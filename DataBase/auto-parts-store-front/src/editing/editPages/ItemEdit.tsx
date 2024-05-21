import React, {useEffect, useState} from "react";
import {Form, Button, Table, Input, Popconfirm, message, Modal, Select} from "antd";

import PostService from "../../postService/PostService";

const ItemEdit: React.FC = () => {
    const [itemsData, setItemsData] = useState([]);
    const [categories, setCategories] = useState([]);
    const [editMode, setEditMode] = useState(false);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [form] = Form.useForm();
    const [currentItem, setCurrentItem] = useState<any>();

    useEffect(() => {
        fetchItems();
        fetchCategories();
    }, []);

    const fetchItems = () => {
        PostService.getRequest(`items/all`).then((response: any) => {
            setItemsData(response.data);
        });
    };

    const fetchCategories = () => {
        PostService.getRequest(`itemCategory/all`).then((response: any) => {
            setCategories(response.data);
        });
    };

    const handleSave = async (values: any) => {
        try {
            const body = {
                name: values.name,
                categoryName: values.categoryName,
                amount: values.amount,
                defectAmount: values.defectAmount,
                price: values.price,
                cellNumber: values.cellNumber,
            };

            if (editMode && currentItem) {
                await PostService.updateRequest(`items/${currentItem.itemId}`, body);
            } else {
                await PostService.addRequest(`items`, body);
            }
            fetchItems();
            resetForm();
        } catch (error) {
            message.error("Failed to save the item.");
        }
    };

    const handleDelete = async (itemId: number) => {
        try {
            await PostService.deleteRequest(`items/${itemId}`);
            message.success("Deleted item.");
            fetchItems();
        } catch (error) {
            message.error("Failed to delete the item.");
        }
    };

    const resetForm = () => {
        setEditMode(false);
        setIsModalVisible(false);
        form.resetFields();
        setCurrentItem(null);
    };

    const handleAdd = () => {
        setEditMode(false);
        setIsModalVisible(true);
    };

    const handleEdit = (record: any) => {
        setEditMode(true);
        setIsModalVisible(true);
        setCurrentItem(record);
        form.setFieldsValue({
            name: record.name,
            categoryName: record.categoryName,
            amount: record.amount,
            defectAmount: record.defectAmount,
            price: record.price,
            cellNumber: record.cellNumber,
        });
    };

    const columns = [
        {
            title: "Наименование",
            dataIndex: "name",
            key: "name",
            sorter: (a: any, b: any) => a.name.localeCompare(b.name),
        },
        {
            title: "Категория",
            dataIndex: "categoryName",
            key: "categoryName"
        },
        {
            title: "Количество",
            dataIndex: "amount",
            key: "amount",
        },
        {
            title: "Дефектное количество",
            dataIndex: "defectAmount",
            key: "defectAmount",
        },
        {
            title: "Цена",
            dataIndex: "price",
            key: "price",
        },
        {
            title: "Номер ячейки",
            dataIndex: "cellNumber",
            key: "cellNumber",
        },
        {
            title: "Действия",
            key: "actions",
            render: (text: any, record: any) => (
                <span>
                    <a onClick={() => handleEdit(record)} style={{marginRight: "10px"}}>Редактировать</a>
                    <Popconfirm
                        title="Вы уверены, что хотите удалить этот элемент?"
                        onConfirm={() => handleDelete(record.itemId)}
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
            <h2 style={{marginBottom: "15px"}}>Детали</h2>
            <Button type="primary" onClick={handleAdd} style={{marginBottom: "15px"}}>
                Добавить
            </Button>
            <Table columns={columns} dataSource={itemsData} rowKey="itemId"/>
            <Modal
                title={editMode ? "Редактировать элемент" : "Добавить элемент"}
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
                        label="Наименование"
                        name="name"
                        rules={[{required: true, message: "Пожалуйста, введите наименование"}]}
                    >
                        <Input/>
                    </Form.Item>
                    <Form.Item
                        label="Категория"
                        name="categoryName"
                        rules={[{required: true, message: "Пожалуйста, выберите категорию"}]}
                    >
                        <Select>
                            {categories.map((category: any) => (
                                <Select.Option key={category.categoryName} value={category.categoryName}>
                                    {category.categoryName}
                                </Select.Option>
                            ))}
                        </Select>
                    </Form.Item>
                    <Form.Item
                        label="Количество"
                        name="amount"
                        rules={[{required: true, message: "Пожалуйста, введите количество"}]}
                    >
                        <Input type="number"/>
                    </Form.Item>
                    <Form.Item
                        label="Дефектное количество"
                        name="defectAmount"
                        rules={[{required: true, message: "Пожалуйста, введите дефектное количество"}]}
                    >
                        <Input type="number"/>
                    </Form.Item>
                    <Form.Item
                        label="Цена"
                        name="price"
                        rules={[{required: true, message: "Пожалуйста, введите цену"}]}
                    >
                        <Input type="number"/>
                    </Form.Item>
                    <Form.Item
                        label="Номер ячейки"
                        name="cellNumber"
                        rules={[{required: true, message: "Пожалуйста, введите номер ячейки"}]}
                    >
                        <Input type="number"/>
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default ItemEdit;
