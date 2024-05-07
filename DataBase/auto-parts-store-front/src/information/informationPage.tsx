import React, {useState} from "react";
import "../information/informationPage.css";
import CashReportQuery from "../querries/CashReportQuery";
import DefectItemsQuery from "../querries/DefectItemsQuery";
import SupplierQuery from "../querries/SupplierQuery";
import SupplierByItemQuery from "../querries/SupplierByItemQuery";
import SupplierByDeliveryQuery from "../querries/SupplierByDeliveryQuery";
import ItemDeliveryPriceInfo from "../querries/ItemDeliveryPriceInfo";

const InformationPage: React.FC = () => {
    const [activeIndexes, setActiveIndexes] = useState<number[]>([]);
    const handleAccordionClick = (index: number) => {
        const indexExists = activeIndexes.includes(index);
        setActiveIndexes((prevIndexes) =>
            indexExists ? prevIndexes.filter((prevIndex) => prevIndex !== index) : [...prevIndexes, index]
        );
    };


    return (
        <div>
            <button
                className={`accordion ${activeIndexes.includes(0) ? "active" : ""}`}
                onClick={() => handleAccordionClick(0)}
            >
                Получить перечень и общее число поставщиков определенной категории
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(0) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <SupplierQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(1) ? "active" : ""}`}
                onClick={() => handleAccordionClick(1)}
            >
                Получить перечень поставщиков поставляющих указанный вид товара
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(1) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <SupplierByItemQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(2) ? "active" : ""}`}
                onClick={() => handleAccordionClick(2)}
            >
                Получить перечень поставщиков поставивших указанный товар в объеме, не менее заданного за определенный
                период
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(2) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <SupplierByDeliveryQuery/>
            </div>


            <button
                className={`accordion ${activeIndexes.includes(3) ? "active" : ""}`}
                onClick={() => handleAccordionClick(3)}
            >
                Получить сведения о конкретном виде деталей: какими поставщиками поставляется, их расценки, время
                поставки.
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(3) ? "block" : "none"}}>
                <ItemDeliveryPriceInfo/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(13) ? "active" : ""}`}
                onClick={() => handleAccordionClick(13)}
            >
                Получить перечень бракованного товара, пришедшего за определенный период
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(13) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <DefectItemsQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(15) ? "active" : ""}`}
                onClick={() => handleAccordionClick(15)}
            >
                Получить кассовый отчет за определенный период
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(15) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <CashReportQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(3) ? "active" : ""}`}
                onClick={() => handleAccordionClick(3)}
            >
                Получить информацию о клиентах
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(3) ? "block" : "none"}}>
                <p>Параметры:</p>
            </div>
        </div>
    );
};

export default InformationPage;
