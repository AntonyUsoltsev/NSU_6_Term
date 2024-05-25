import React, {useEffect, useState} from 'react';
import {Button, Select, Spin, Table} from 'antd';
import './Catalog.css';
import PostService from '../postService/PostService';

const {Option} = Select;

const CatalogPage: React.FC = () => {
    const [loading, setLoading] = useState(true);
    const [catalog, setCatalog] = useState([]);

    useEffect(() => {
        PostService.getCatalog().then((response: any) => {
            setCatalog(response.data);
            setLoading(false);
        });
    }, []);

    const uniqueCategories = Array.from(new Set(catalog.map((item: any) => item.categoryName)));
    const uniqueSuppliers = Array.from(new Set(catalog.flatMap((item: any) => item.supplierItemInfos.map((supplierItem: any) => supplierItem.supplierName))));

    const columns = [
        {
            title: 'Название детали',
            dataIndex: 'itemName',
            key: 'itemName',
        },
        {
            title: 'Категория',
            dataIndex: 'categoryName',
            key: 'categoryName',
            filterDropdown: ({setSelectedKeys, selectedKeys, confirm, clearFilters}: any) => (
                <div style={{padding: 8}}>
                    <Select
                        placeholder="Выберите категорию"
                        value={selectedKeys[0]}
                        onChange={(value) => setSelectedKeys(value ? [value] : [])}
                        style={{width: 190, marginBottom: 8, display: 'block'}}
                    >
                        {uniqueCategories.map((category, index) => (
                            <Option key={index} value={category}>
                                {category}
                            </Option>
                        ))}
                    </Select>
                    <Button
                        type="primary"
                        onClick={() => confirm()}
                        size="small"
                        style={{width: 90, marginRight: 8}}
                    >
                        Поиск
                    </Button>
                    <Button onClick={() => clearFilters()} size="small" style={{width: 90}}>
                        Сбросить
                    </Button>
                </div>
            ),
            onFilter: (value: any, record: any) => record.categoryName.toLowerCase().includes(value.toLowerCase()),
        },
        {
            title: 'Количество',
            dataIndex: 'amount',
            key: 'amount',
        },
        {
            title: 'Цена',
            dataIndex: 'price',
            key: 'price',
            sorter: (a: any, b: any) => a.price - b.price,
        },
        {
            title: 'Имя поставщика',
            dataIndex: 'supplierName',
            key: 'supplierName',
            filterDropdown: ({setSelectedKeys, selectedKeys, confirm, clearFilters}: any) => (
                <div style={{padding: 8}}>
                    <Select
                        placeholder="Выберите поставщика"
                        value={selectedKeys[0]}
                        onChange={(value) => setSelectedKeys(value ? [value] : [])}
                        style={{width: 190, marginBottom: 8, display: 'block'}}
                    >
                        {uniqueSuppliers.map((supplier, index) => (
                            <Option key={index} value={supplier}>
                                {supplier}
                            </Option>
                        ))}
                    </Select>
                    <Button
                        type="primary"
                        onClick={() => confirm()}
                        size="small"
                        style={{width: '49%', marginRight: 8}}
                    >
                        Поиск
                    </Button>
                    <Button onClick={() => clearFilters()} size="small" style={{width: '49%'}}>
                        Сбросить
                    </Button>
                </div>
            ),
            onFilter: (value: any, record: any) => record.supplierName.toLowerCase().includes(value.toLowerCase()),
        },
    ];

    // Data for the table
    const data = catalog.reduce((acc: any, item: any) => {
        const {itemName, categoryName, supplierItemInfos} = item;
        // Map through supplierItemInfos and add a row for each item
        supplierItemInfos.forEach((supplierItem: any) => {
            acc.push({
                key: `${itemName}-${supplierItem.supplierName}`, // Unique key for each row
                itemName,
                categoryName,
                amount: supplierItem.amount,
                price: supplierItem.price,
                supplierName: supplierItem.supplierName,
            });
        });
        return acc;
    }, []);

    return (
        <div>
            <header className="catalog-header">Каталог запчастей</header>

            {loading ? (
                <Spin size="large"/>
            ) : (
                <Table
                    columns={columns}
                    dataSource={data}
                    pagination={{pageSize: 20}}
                />
            )}
        </div>
    );
};

export default CatalogPage;
