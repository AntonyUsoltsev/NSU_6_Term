import React, {useState} from "react";
import "../information/informationPage.css";
import CashReportQuery from "./querries/CashReportQuery";
import DefectItemsQuery from "./querries/DefectItemsQuery";
import SupplierQuery from "./querries/SupplierQuery";
import SupplierByItemQuery from "./querries/SupplierByItemQuery";
import SupplierByDeliveryQuery from "./querries/SupplierByDeliveryQuery";
import ItemDeliveryPriceInfo from "./querries/ItemDeliveryPriceInfo";
import CustomerByItemQuery from "./querries/CustomerByItemQuery";
import CustomerByItemWithAmountQuery from "./querries/CustomerByItemWithAmontQuery";
import ItemsInfoQuery from "./querries/ItemsInfoQuery";
import TopTenItemsQuery from "./querries/TopTenItemsQuery";
import RealisedItemsByDayQuery from "./querries/RealisedItemsByDayQuery";
import SellingSpeedQuery from "./querries/SellingSpeedQuery";
import InventoryListQuery from "./querries/InventoryListQuery";
import StoreCapacityQuery from "./querries/StoreCapacityQuery";
import AverageSellQuery from "./querries/AverageSellQuery";

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
            {params}
            {content}
        </div>
    </>
);

const InformationPage: React.FC = () => {
    const [activeIndexes, setActiveIndexes] = useState<number[]>([]);

    const handleAccordionClick = (index: number) => {
        const indexExists = activeIndexes.includes(index);
        setActiveIndexes((prevIndexes) =>
            indexExists ? prevIndexes.filter((prevIndex) => prevIndex !== index) : [...prevIndexes, index]
        );
    };

    const accordionItemsData = [
        {
            title: "1. Получить перечень и общее число поставщиков определенной категории",
            params: <p style={{marginBottom: "20px"}}>Параметры:</p>,
            content: <SupplierQuery/>,
            index: 1,
        },
        {
            title: "2. Получить перечень поставщиков поставляющих указанный вид товара",
            params: <p style={{marginBottom: "20px"}}>Параметры:</p>,
            content: <SupplierByItemQuery/>,
            index: 2,
        },
        {
            title: "3. Получить перечень поставщиков поставивших указанный товар в объеме, не менее заданного за определенный период",
            params: <p style={{marginBottom: "20px"}}>Параметры:</p>,
            content: <SupplierByDeliveryQuery/>,
            index: 3,
        },
        {
            title: "4. Получить сведения о конкретном виде деталей: какими поставщиками поставляется, их расценки, время поставки.",
            content: <ItemDeliveryPriceInfo/>,
            index: 4,
        },
        {
            title: "5. Получить перечень и общее число покупателей, купивших указанный вид товара за некоторый период не менее указанного объема",
            params: <p style={{marginBottom: "20px"}}>Параметры:</p>,
            content: <CustomerByItemQuery/>,
            index: 5,
        },
        {
            title: "6. Получить перечень и общее число покупателей, купивших указанный вид товара за некоторый период",
            params: <p style={{marginBottom: "20px"}}>Параметры:</p>,
            content: <CustomerByItemWithAmountQuery/>,
            index: 6,
        },
        {
            title: "7. Получить перечень, объем и номер ячейки для всех деталей, хранящихся на складе",
            content: <ItemsInfoQuery/>,
            index: 7,
        },
        {
            title: "8. Получить десять самых продаваемых деталей",
            content: <TopTenItemsQuery/>,
            index: 8,
        },
        {
            title: "10. Получить среднее число продаж на месяц по любому виду деталей",
            content: <AverageSellQuery/>,
            index: 10,
        },
        {
            title: "14. Получить перечень бракованного товара, пришедшего за определенный период",
            params: <p style={{marginBottom: "20px"}}>Параметры:</p>,
            content: <DefectItemsQuery/>,
            index: 14,
        },
        {
            title: "15. Получить перечень, общее количество и стоимость товара, реализованного за конкретный день.",
            params: <p style={{marginBottom: "20px"}}>Параметры:</p>,
            content: <RealisedItemsByDayQuery/>,
            index: 15,
        },
        {
            title: "16. Получить кассовый отчет за определенный период",
            params: <p style={{marginBottom: "20px"}}>Параметры:</p>,
            content: <CashReportQuery/>,
            index: 16,
        },
        {
            title: "17. Получить инвентаризационную ведомость",
            content: <InventoryListQuery/>,
            index: 17,
        },
        {
            title: "18. Получить скорость оборота денежных средств, вложенных в товар",
            content: <SellingSpeedQuery/>,
            index: 18,
        },
        {
            title: "19. Подсчитать сколько пустых ячеек на складе и сколько он сможет вместить товара",
            content: <StoreCapacityQuery/>,
            index: 19,
        },
    ];


    return (
        <div>
            {accordionItemsData.map((item) => (
                <AccordionItem
                    key={item.index}
                    title={item.title}
                    content={item.content}
                    activeIndexes={activeIndexes}
                    index={item.index}
                    params={item.params}
                    handleAccordionClick={handleAccordionClick}
                />
            ))}
        </div>
    );
};

export default InformationPage;
