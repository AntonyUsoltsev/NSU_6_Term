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

    static async getRequest(url: any) {
        console.log(url)
        try {
            const value = await axios.get(url);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async getCatalog() {
        return this.getRequest("http://localhost:8080/AutoPartsStore/api/items/catalog");
    }

    static async getCashReport(startDate: string, endDate: string) {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/transactions/cashReport?from=${startDate}&to=${endDate}`);
    }

    static async getDefectItems(startDate: string, endDate: string) {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/items/defect?from=${startDate}&to=${endDate}`);
    }

    static async getSuppliersByType(category: string) {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/suppliers?type=${category}`);
    }

    static async getSuppliersByItemType(category: string) {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/suppliers/itemCategory?category=${category}`);
    }

    static async getSuppliersByDelivery(startDate: string, endDate: string, amount: string, itemName: string) {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/suppliers/delivery?from=${startDate}&to=${endDate}&amount=${amount}&item=${itemName}`);
    }

    static async getItemsDeliveryPriceInfo() {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/items/deliveryPrice`);
    }

    static async getCustomerByItem(startDate: string, endDate: string, itemName: string) {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/customers/byItem?from=${startDate}&to=${endDate}&item=${itemName}`);
    }

    static async getCustomerByItemWithAmount(startDate: string, endDate: string, amount: string, itemName: string) {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/customers/byItemWithAmount?from=${startDate}&to=${endDate}&amount=${amount}&item=${itemName}`);
    }

    static async getItemsInfo() {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/items/info`);
    }

    static async getItemsTopTen() {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/items/top`);
    }

    static async getRealisedItems(day: string) {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/transactions/realised?date=${day}`);
    }

    static async getSellingSpeed() {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/transactions/sellSpeed`);
    }

    static async getInventoryList() {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/items/all`);
    }

    static async getStoreCapacity() {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/items/storeCapacity`);
    }

    static async getAverageSell() {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/transactions/averageSell`);
    }

    static async getSupplierTypes() {
        return this.getRequest(`http://localhost:8080/AutoPartsStore/api/supplierType/all`);
    }
}