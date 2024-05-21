import axios from "axios";

export default class PostService {

    static requestPrefix = `http://localhost:8080/AutoPartsStore/api/`

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
        console.log("Get request to: " + url)
        try {
            const value = await axios.get(this.requestPrefix + url);
            console.log("Get answer from " + url + " : ")
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
        }
    }

    static async deleteRequest(url: any) {
        console.log("Delete request to: " + url)
        // try {
        //     const value = await axios.delete(this.requestPrefix + url);
        //     console.log("Del answer from " + url + " : " + value)
        //     return value;
        // } catch (error) {
        //     this.errorHandler(error);
        //     throw error;
        // }
    }

    static async addRequest(url: any, body: {}) {
        console.log("Add request to: " + url + "body: ");
        console.log(body);
        try {
            const value = await axios.post(this.requestPrefix + url, body)
            console.log("Add answer from " + url + " : ")
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
            throw error;
        }
    }

    static async updateRequest(url: any, body: {}) {
        console.log("Update request to: " + url + " body: ");
        console.log(body);
        try {
            const value = await axios.patch(this.requestPrefix + url, body)
            console.log("Update answer from " + url + " : ")
            console.log(value)
            return value;
        } catch (error) {
            this.errorHandler(error);
            throw error;
        }
    }


    static async getCatalog() {
        return this.getRequest("items/catalog");
    }

    static async getCashReport(startDate: string, endDate: string) {
        return this.getRequest(`transactions/cashReport?from=${startDate}&to=${endDate}`);
    }

    static async getDefectItems(startDate: string, endDate: string) {
        return this.getRequest(`items/defect?from=${startDate}&to=${endDate}`);
    }

    static async getSuppliersByType(category: string) {
        return this.getRequest(`suppliers?type=${category}`);
    }

    static async getSuppliersByItemType(category: string) {
        return this.getRequest(`suppliers/itemCategory?category=${category}`);
    }

    static async getSuppliersByDelivery(startDate: string, endDate: string, amount: string, itemName: string) {
        return this.getRequest(`suppliers/delivery?from=${startDate}&to=${endDate}&amount=${amount}&item=${itemName}`);
    }

    static async getItemsDeliveryPriceInfo() {
        return this.getRequest(`items/deliveryPrice`);
    }

    static async getCustomerByItem(startDate: string, endDate: string, itemName: string) {
        return this.getRequest(`customers/byItem?from=${startDate}&to=${endDate}&item=${itemName}`);
    }

    static async getCustomerByItemWithAmount(startDate: string, endDate: string, amount: string, itemName: string) {
        return this.getRequest(`customers/byItemWithAmount?from=${startDate}&to=${endDate}&amount=${amount}&item=${itemName}`);
    }

    static async getItemsInfo() {
        return this.getRequest(`items/info`);
    }

    static async getItemsTopTen() {
        return this.getRequest(`items/top`);
    }

    static async getRealisedItems(day: string) {
        return this.getRequest(`transactions/realised?date=${day}`);
    }

    static async getSellingSpeed() {
        return this.getRequest(`transactions/sellSpeed`);
    }

    static async getInventoryList() {
        return this.getRequest(`items/all`);
    }

    static async getStoreCapacity() {
        return this.getRequest(`items/storeCapacity`);
    }

    static async getAverageSell() {
        return this.getRequest(`transactions/averageSell`);
    }

}