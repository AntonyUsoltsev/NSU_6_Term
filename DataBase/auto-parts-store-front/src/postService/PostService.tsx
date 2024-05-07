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


    static async  getItemsDeliveryPriceInfo() {
        try {
            const value = await axios.get(`http://localhost:8080/AutoPartsStore/api/items/deliveryPrice`);
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }
}