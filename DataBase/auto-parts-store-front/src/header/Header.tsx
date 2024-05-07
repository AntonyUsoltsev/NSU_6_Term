import React, {useState} from "react";
import {FaShoppingCart} from "react-icons/fa";
import {useHistory} from "react-router-dom";
import "../header/Header.css"

const Header = () => {

    const history = useHistory();

    let [cartOpen, setCartOpen] = useState(false);
    const handleButtonClick = (route: any) => {
        history.push(route);
        window.location.reload();
    };
    return (
        <header>
            <div>
                <span className='logo' onClick={() => handleButtonClick("/")}> Auto Parts Shop</span>

                <ul className='nav'>
                    <li onClick={() => handleButtonClick("/catalog")}>Каталог</li>
                    <li onClick={() => handleButtonClick("/info")}>Информация</li>
                    <li onClick={() => handleButtonClick("/editing")}>Редактирование</li>
                    <li onClick={() => handleButtonClick("/contacts")}>Контакты</li>
                </ul>
                <FaShoppingCart onClick={() => setCartOpen(cartOpen = !cartOpen)}
                                className={`shop-cart-button ${cartOpen && 'active'}`}/>

            </div>
            <div className='presentation'></div>
        </header>
    )
}

export default Header;