import * as actionTypes from "./actionTypes";
import axios from "axios";

export const user_item_tool = (titleid, category) => {
    return {
        type: actionTypes.USER_ITEM,
        titleid: titleid,
        category: category
    };
};

export const user_cat_tool = (title) => {
    return {
        type: actionTypes.USER_CAT,
        title: title
    };
};

export const api_request = (url, text) => {
    console.log(`http://127.0.0.1:8000/{url}`)
    url = url.replace(/\s/g, "");
    url = "http://127.0.0.1:8000/api/" + url + "/" + `?text=${text}`;
    console.log(url, text)
    debugger
    return dispatch => {
        debugger
        // axios.defaults.headers = {
        //     "Content-Type": "application/json",
        //     Authorization: `Token ${token}`
        // };
        axios
            .get(url)
            .then(res => {
                debugger
            })
            .catch(err => {
                debugger
            });
    };
};
