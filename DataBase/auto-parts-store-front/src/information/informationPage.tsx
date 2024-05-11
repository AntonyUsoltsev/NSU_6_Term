import React, {useState} from "react";
import "../information/informationPage.css";
import CashReportQuery from "../querries/CashReportQuery";
import DefectItemsQuery from "../querries/DefectItemsQuery";
import SupplierQuery from "../querries/SupplierQuery";
import SupplierByItemQuery from "../querries/SupplierByItemQuery";
import SupplierByDeliveryQuery from "../querries/SupplierByDeliveryQuery";
import ItemDeliveryPriceInfo from "../querries/ItemDeliveryPriceInfo";
import CustomerByItemQuery from "../querries/CustomerByItemQuery";
import CustomerByItemWithAmountQuery from "../querries/CustomerByItemWithAmontQuery";
import ItemsInfoQuery from "../querries/ItemsInfoQuery";
import TopTenItemsQuery from "../querries/TopTenItemsQuery";
import RealisedItemsByDayQuery from "../querries/RealisedItemsByDayQuery";
import SellingSpeedQuery from "../querries/SellingSpeedQuery";
import InventoryListQuery from "../querries/InventoryListQuery";
import StoreCapacityQuery from "../querries/StoreCapacityQuery";
import AverageSellQuery from "../querries/AverageSellQuery";

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
                className={`accordion ${activeIndexes.includes(1) ? "active" : ""}`}
                onClick={() => handleAccordionClick(1)}
            >
                1. Получить перечень и общее число поставщиков определенной категории
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(1) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <SupplierQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(2) ? "active" : ""}`}
                onClick={() => handleAccordionClick(2)}
            >
                2. Получить перечень поставщиков поставляющих указанный вид товара
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(2) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <SupplierByItemQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(3) ? "active" : ""}`}
                onClick={() => handleAccordionClick(3)}
            >
                3. Получить перечень поставщиков поставивших указанный товар в объеме, не менее заданного за
                определенный
                период
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(3) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <SupplierByDeliveryQuery/>
            </div>


            <button
                className={`accordion ${activeIndexes.includes(4) ? "active" : ""}`}
                onClick={() => handleAccordionClick(4)}
            >
                4. Получить сведения о конкретном виде деталей: какими поставщиками поставляется, их расценки, время
                поставки.
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(4) ? "block" : "none"}}>
                <ItemDeliveryPriceInfo/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(5) ? "active" : ""}`}
                onClick={() => handleAccordionClick(5)}
            >
                5. Получить перечень и общее число покупателей, купивших указанный вид товара за некоторый период не
                менее указанного объема
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(5) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <CustomerByItemQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(6) ? "active" : ""}`}
                onClick={() => handleAccordionClick(6)}
            >
                6. Получить перечень и общее число покупателей, купивших указанный вид товара за некоторый период
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(6) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <CustomerByItemWithAmountQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(7) ? "active" : ""}`}
                onClick={() => handleAccordionClick(7)}
            >
                7. Получить перечень, объем и номер ячейки для всех деталей, хранящихся на складе
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(7) ? "block" : "none"}}>
                <ItemsInfoQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(8) ? "active" : ""}`}
                onClick={() => handleAccordionClick(8)}
            >
                8. Получить десять самых продаваемых деталей
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(8) ? "block" : "none"}}>
                <TopTenItemsQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(10) ? "active" : ""}`}
                onClick={() => handleAccordionClick(10)}
            >
                10. Получить среднее число продаж на месяц по любому виду деталей
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(10) ? "block" : "none"}}>
                <AverageSellQuery/>
            </div>


            <button
                className={`accordion ${activeIndexes.includes(14) ? "active" : ""}`}
                onClick={() => handleAccordionClick(14)}
            >
                14. Получить перечень бракованного товара, пришедшего за определенный период
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(14) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <DefectItemsQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(15) ? "active" : ""}`}
                onClick={() => handleAccordionClick(15)}
            >
                15. Получить перечень, общее количество и стоимость товара, реализованного за конкретный день.
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(15) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <RealisedItemsByDayQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(16) ? "active" : ""}`}
                onClick={() => handleAccordionClick(16)}
            >
                16. Получить кассовый отчет за определенный период
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(16) ? "block" : "none"}}>
                <p style={{marginBottom: "20px"}}>Параметры:</p>
                <CashReportQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(17) ? "active" : ""}`}
                onClick={() => handleAccordionClick(17)}
            >
                17. Получить инвентаризационную ведомость
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(17) ? "block" : "none"}}>
                <InventoryListQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(18) ? "active" : ""}`}
                onClick={() => handleAccordionClick(18)}
            >
                18. Получить скорость оборота денежных средств, вложенных в товар
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(18) ? "block" : "none"}}>
                <SellingSpeedQuery/>
            </div>

            <button
                className={`accordion ${activeIndexes.includes(19) ? "active" : ""}`}
                onClick={() => handleAccordionClick(19)}
            >
                19. Подсчитать сколько пустых ячеек на складе и сколько он сможет вместить товара
                <i className="fas fa-angle-down"></i>
            </button>
            <div className="panel" style={{display: activeIndexes.includes(19) ? "block" : "none"}}>
                <StoreCapacityQuery/>
            </div>

        </div>
    );
};

export default InformationPage;
