import React from 'react';
import {Button} from 'antd';
import {useHistory} from "react-router-dom";
// @ts-ignore
import logoImage from '../images/shopLogo.jpg';
import './MainMenu.css';

const MainMenu = () => {
    const history = useHistory();

    const handleButtonClick = (route: any) => {
        history.push(route);
        window.location.reload();
    };

    return (
        <div className="mainMenuContainer">
            <Button className="logoButton" onClick={() => handleButtonClick("/")}>
                <img src={logoImage} alt="Logo" className="logoImage"/>
            </Button>
            <Button className="catalogButton" onClick={() => handleButtonClick("/catalog")}>
                <span className="burgerIcon">sd</span> Каталог
            </Button>
            <Button className="cartButton" onClick={() => handleButtonClick("/cart")}>
                Корзина
            </Button>
            <Button className="infoButton" onClick={() => handleButtonClick("/info")}>
                Информация
            </Button>
        </div>
    );
};

export default MainMenu;
