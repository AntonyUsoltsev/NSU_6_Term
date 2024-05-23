import React, {useState} from "react";
import "../information/informationPage.css";
import AverageSellQuery from "../information/querries/AverageSellQuery";
import DefectItemsQuery from "../information/querries/DefectItemsQuery";
import SupplierTypeEdit from "./editPages/SupplierTypeEdit";
import ItemCategoryEdit from "./editPages/ItemCategoryEdit";
import TransactionTypeEdit from "./editPages/TransactionTypeEdit";
import SupplierEdit from "./editPages/SupplierEdit";
import CustomerEdit from "./editPages/CustomerEdit";
import CashierEdit from "./editPages/CashierEdit";
import ItemEdit from "./editPages/ItemEdit";
import DeliveryEdit from "./editPages/DeliveryEdit";

const AccordionItem = ({title, content, activeIndexes, index, handleAccordionClick, params}: any) => (
    <>
        <button
            className={`accordion ${activeIndexes.includes(index) ? "active" : ""}`}
            onClick={() => handleAccordionClick(index)}
        >
            {title}
            <i className="fas fa-angle-down"></i>
        </button>
        <div className="panel" style={{display: activeIndexes.includes(index) ? "block" : "none"}}>
            {content}
        </div>
    </>
);

const EditPage: React.FC = () => {
    const [activeIndexes, setActiveIndexes] = useState<number[]>([]);
    const accordionItemsData = [
        {
            title: "Редактирование типов поставщиков",
            content: <SupplierTypeEdit/>,
            index: 1,
        },
        {
            title: "Редактирование поставщиков",
            content: <SupplierEdit/>,
            index: 2,
        },
        {
            title: "Редактирование поставок",
            content: <DeliveryEdit/>,
            index: 3,
        },
        {
            title: "Редактирование типов деталей",
            content: <ItemCategoryEdit/>,
            index: 4,
        },
        {
            title: "Редактирование деталей",
            content: <ItemEdit/>,
            index: 5,
        },
        {
            title: "Редактирование покупателей",
            content: <CustomerEdit/>,
            index: 6,
        },
        {
            title: "Редактирование кассиров",
            content: <CashierEdit/>,
            index: 7,
        },
        {
            title: "Редактирование типов транзакций",
            content: <TransactionTypeEdit/>,
            index: 8,
        },
        {
            title: "Редактирование транзакций",
            content: <div></div>,
            index: 9,
        },
        {
            title: "Редактирование заказов",
            content: <div></div>,
            index: 10,
        }
    ];
    const handleAccordionClick = (index: number) => {
        const indexExists = activeIndexes.includes(index);
        setActiveIndexes((prevIndexes) =>
            indexExists ? prevIndexes.filter((prevIndex) => prevIndex !== index) : [...prevIndexes, index]
        );
    };
    return (
        <div>
            {accordionItemsData.map((item) => (
                <AccordionItem
                    key={item.index}
                    title={item.title}
                    content={item.content}
                    activeIndexes={activeIndexes}
                    index={item.index}
                    handleAccordionClick={handleAccordionClick}
                />
            ))}
        </div>
    );
}

export default EditPage;