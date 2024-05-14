import React, {useEffect, useState} from "react";
import {Form, Button, Table, Input, Popconfirm, message} from "antd";
import PostService from "../../postService/PostService";

const SupplierTypeEdit: React.FC = () => {
    const [supplierTypesData, setSupplierTypesData] = useState([]);
    const [editMode, setEditMode] = useState(false);
    const [typeName, setTypeName] = useState("");
    const [selectedSupplierTypeId, setSelectedSupplierTypeId] = useState(null);

    useEffect(() => {
        fetchSupplierTypes();
    }, []);

    const fetchSupplierTypes = () => {
        PostService.getSupplierTypes().then((response: any) => {
            setSupplierTypesData(response.data);
        });
    };

    const handleSave = async () => {
        try {
            if (editMode) {
                await PostService.updateSupplierType(typeName, selectedSupplierTypeId);
            } else {
                await PostService.addSupplierType(typeName);
            }
            fetchSupplierTypes();
            setEditMode(false);
            setTypeName("");
            setSelectedSupplierTypeId(null);
        } catch (error) {
            message.error("Failed to save the supplier type.");
        }
    };

    const handleDelete = async (supplierTypeId: number) => {
        try {
            PostService.deleteSupplierType(supplierTypeId).then((respone: any) => {
                    fetchSupplierTypes()
                }
            );

        } catch (error) {
            message.error("Failed to delete the supplier type.");
        }
    };

    const columns = [
        {
            title: "Тип поставщика",
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
                        setSelectedSupplierTypeId(record.typeId);
                    }}>Редактировать</a>
                    <Popconfirm
                        title="Вы уверены, что хотите удалить этот тип поставщика?"
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

    return (
        <div>
            <h2 style={{marginBottom: "15px"}}>Категории поставщиков</h2>
            <Form layout="inline">
                <Form.Item label="Тип поставщика">
                    <Input value={typeName} onChange={(e) => setTypeName(e.target.value)}/>
                </Form.Item>
                <Form.Item>
                    <Button type="primary" onClick={handleSave}>
                        {editMode ? "Сохранить" : "Добавить"}
                    </Button>
                </Form.Item>
            </Form>
            <Table columns={columns} dataSource={supplierTypesData}/>
        </div>
    );
};

export default SupplierTypeEdit;
