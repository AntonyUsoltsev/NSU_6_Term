import React from 'react';
import './App.css';
import {BrowserRouter, BrowserRouter as Router, Route, Switch} from 'react-router-dom';
import catalogPage from "./catalog/CatalogPage";
import Header from "./header/Header";
import contactPage from "./contacts/ContactPage";

class App extends React.Component {
     render() {
        return (
            <div className="wrapper">
                <Header></Header>
                <BrowserRouter>
                    {/*<AuthButtons/>*/}
                    {/*<MainMenu/>*/}
                    <Switch>
                        {/*<Route path="/student_compass/:university/:course/:subject" component={BookListPage}/>*/}
                        <Route path="/catalog" component={catalogPage}/>
                        <Route path="/contacts" component={contactPage}/>
                    </Switch>
                </BrowserRouter>
            </div>
        );
    }
}

const Root = () => {
    return (
        <Router basename={process.env.PUBLIC_URL}>
            <App/>
        </Router>
    );
}

export default Root;
