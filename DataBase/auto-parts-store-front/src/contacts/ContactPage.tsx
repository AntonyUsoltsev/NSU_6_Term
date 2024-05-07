import React from "react";
import "../contacts/contactPage.css";

const ContactPage: React.FC = () => {

    return (
        <div>
            <header style={{
                fontSize: '24px',
                fontWeight: 'bold',
            }}
            >
                Контакты:
            </header>
            <div>
                <a
                    href="https://t.me/malignantt"
                    target="_blank"
                    rel="noopener noreferrer"
                    className={'link'}
                >
                    Напишите мне в Telegram
                </a>
            </div>
        </div>
    );
}

export default ContactPage;
