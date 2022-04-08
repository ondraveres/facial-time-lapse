//import html from './index.html';
import React from "react";
import ReactDOM from "react-dom";
import axios from 'axios';

import App from "./components/app";
import "./components/app.css";
import "./style.css"

import "bootstrap/dist/css/bootstrap.css";

const ChildElement = React.createRef();

document.addEventListener('visibilitychange', function logData() {
    const childelement = ChildElement.current;
    const paths = childelement.state.items.map(item => item.path);
    const url = 'http://halmos.felk.cvut.cz:5000/deleteImagesAPI'

    console.log(document.visibilityState)

    if (document.visibilityState === 'hidden') {
        axios.post(url, paths)
    }
});



ReactDOM.render(<App ref={ChildElement} />, document.getElementById("root"));
