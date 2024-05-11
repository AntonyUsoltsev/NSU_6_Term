import axios from "axios";

export default class PostService {
    static errorHandler(error: any) {
        if (error.response) {
            console.error(error.response.data);
            console.error(error.response.status);
            console.error(error.response.headers);
        } else if (error.request) {
            console.error(error.request);
        } else {
            console.error('Error', error.message);
        }
    }

    static async getCatalog(tableId: any) {
        console.log("tableId" + tableId)
        try {
            const value = await axios.get("http://localhost:8080/AutoPartsStore/api/items/catalog");
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getCategories() {
        try {
            const value = await axios.get("http://localhost:8080/AutoPartsStore/api/supplierType/all");
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getCashReport(startDate: any, endDate: any) {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/transactions/cashReport?from=${startDate}&to=${endDate}`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getDefectItems(startDate: any, endDate: any) {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/items/defect?from=${startDate}&to=${endDate}`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getSuppliersByType(category: any) {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/suppliers?type=${category}`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getSuppliersByItemType(category: any) {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/suppliers/itemCategory?category=${category}`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getSuppliersByDelivery(startDate: any, endDate: any, amount: any, itemName: any) {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/suppliers/delivery?from=${startDate}&to=${endDate}&amount=${amount}&item=${itemName}`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }


    static async getItemsDeliveryPriceInfo() {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/items/deliveryPrice`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getCustomerByItem(startDate: any, endDate: any, itemName: any) {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/customers/byItem?from=${startDate}&to=${endDate}&item=${itemName}`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getCustomerByItemWithAmount(startDate: any, endDate: any, amount: any, itemName: any) {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/customers/byItemWithAmount?from=${startDate}&to=${endDate}&amount=${amount}&item=${itemName}`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getItemsInfo() {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/items/info`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getItemsTopTen() {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/items/top`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getRealisedItems(day: any) {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/transactions/realised?date=${day}`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getSellingSpeed() {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/transactions/sellSpeed`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getInventoryList() {
        try {
            const value = await axios.get(` http://localhost:8080/AutoPartsStore/api/items/all`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getStoreCapacity() {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/items/storeCapacity`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getAverageSell() {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/transactions/averageSell`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }
}