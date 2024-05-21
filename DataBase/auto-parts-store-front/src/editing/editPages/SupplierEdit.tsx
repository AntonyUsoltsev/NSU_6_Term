import React, {useEffect, useState} from "react";
import {Form, Button, Table, Input, Popconfirm, message, Modal, Select} from "antd";

import PostService from "../../postService/PostService";

const SupplierEdit: React.FC = () => {
    const [suppliersData, setSuppliersData] = useState([]);
    const [typesData, setTypesData] = useState([]);
    const [editMode, setEditMode] = useState(false);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [form] = Form.useForm();
    const [currentSupplier, setCurrentSupplier] = useState<any>();

    useEffect(() => {
        fetchSuppliers();
        fetchSupplierTypes();
    }, []);

    const fetchSuppliers = () => {
        PostService.getRequest(`suppliers/all`).then((response: any) => {
            setSuppliersData(response.data);
        });
    };

    const fetchSupplierTypes = () => {
        PostService.getRequest(`supplierType/all`).then((response: any) => {
            setTypesData(response.data);
        });
    };

    const handleSave = async (values: any) => {
        try {
            const body = {
                name: values.name,
                documents: values.documents,
                typeName: values.typeName,
                garanty: values.garanty,
            };

            if (editMode && currentSupplier) {
                await PostService.updateRequest(`suppliers/${currentSupplier.supplierId}`, body);
            } else {
                await PostService.addRequest(`suppliers`, body);
            }
            fetchSuppliers();
            resetForm();
        } catch (error) {
            message.error("Failed to save the supplier.");
        }
    };

    const handleDelete = async (supplierId: number) => {
        try {
            await PostService.deleteRequest(`suppliers/${supplierId}`);
            message.success("Deleted supplier.");
            fetchSuppliers();
        } catch (error) {
            message.error("Failed to delete the supplier.");
        }
    };

    const resetForm = () => {
        setEditMode(false);
        setIsModalVisible(false);
        form.resetFields();
        setCurrentSupplier(null);
    };

    const handleAdd = () => {
        setEditMode(false);
        setIsModalVisible(true);
    };

    const handleEdit = (record: any) => {
        setEditMode(true);
        setIsModalVisible(true);
        setCurrentSupplier(record);
        form.setFieldsValue({
            name: record.name,
            documents: record.documents,
            typeName: record.typeName,
            garanty: record.garanty,
        });
    };

    const columns = [
        {
            title: "Название",
            dataIndex: "name",
            key: "name",
        },
        {
            title: "Документы",
            dataIndex: "documents",
            key: "documents",
        },
        {
            title: "Тип",
            dataIndex: "typeName",
            key: "typeName",
        },
        {
            title: "Гарантия",
            dataIndex: "garanty",
            key: "garanty",
            render: (garanty: boolean) => (garanty ? "Есть" : "Нет"),
        },
        {
            title: "Действия",
            key: "actions",
            render: (text: any, record: any) => (
                <span>
                    <a onClick={() => handleEdit(record)} style={{marginRight: "10px"}}>Редактировать</a>
                    <Popconfirm
                        title="Вы уверены, что хотите удалить этого поставщика?"
                        onConfirm={() => handleDelete(record.supplierId)}
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
            <h2 style={{marginBottom: "15px"}}>Поставщики</h2>
            <Button type="primary" onClick={handleAdd} style={{marginBottom: "15px"}}>
                Добавить
            </Button>
            <Table columns={columns} dataSource={suppliersData}/>
            <Modal
                title={editMode ? "Редактировать поставщика" : "Добавить поставщика"}
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
                        label="Имя поставщика"
                        name="name"
                        rules={[{required: true, message: "Пожалуйста, введите имя поставщика"}]}
                    >
                        <Input />
                    </Form.Item>
                    <Form.Item
                        label="Документы"
                        name="documents"
                    >
                        <Input />
                    </Form.Item>
                    <Form.Item
                        label="Тип"
                        name="typeName"
                        rules={[{required: true, message: "Пожалуйста, выберите тип"}]}
                    >
                        <Select>
                            {typesData.map((type: any) => (
                                <Select.Option key={type.typeId} value={type.typeName}>
                                    {type.typeName}
                                </Select.Option>
                            ))}
                        </Select>
                    </Form.Item>
                    <Form.Item
                        label="Гарантия"
                        name="garanty"
                        valuePropName="checked"
                    >
                        <Input type="checkbox" />
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default SupplierEdit;
