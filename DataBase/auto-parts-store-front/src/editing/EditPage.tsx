import React, {useState} from "react";
import "../information/informationPage.css";
import AverageSellQuery from "../information/querries/AverageSellQuery";
import DefectItemsQuery from "../information/querries/DefectItemsQuery";
import SupplierTypeEdit from "./editPages/SupplierTypeEdit";

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
            content: <div></div>,
            index: 2,
        },
        {
            title: "Редактирование поставок",
            content: <div></div>,
            index: 3,
        },
        {
            title: "Редактирование типов деталей",
            content: <div></div>,
            index: 4,
        },
        {
            title: "Редактирование деталей",
            content: <div></div>,
            index: 5,
        },
        {
            title: "Редактирование покупателей",
            content: <div></div>,
            index: 6,
        },
        {
            title: "Редактирование кассиров",
            content: <div></div>,
            index: 7,
        },
        {
            title: "Редактирование типов транзакций",
            content: <div></div>,
            index: 8,
        },
        {
            title: "Редактирование транзакций",
            content: <div></div>,
            index: 9,
        },
        {
            title: "Редактирование заказов",
            content: <DefectItemsQuery/>,
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