import * as actionTypes from "./actionTypes";
import { URL } from "./config";
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

export const responseSuccess = (res) => {
    return {
        type: actionTypes.RESPONSE_SUCCESS,
        data: res.data,
        reserror: "",
    }
}

export const responseFailed = (error) => {
    return {
        type: actionTypes.RESPONSE_FAILED,
        data: "",
        reserror: error
    }
}

export const api_request = (url, text) => {
    console.log(URL, url)
    url = url.replace(/\s/g, "");
    url = URL + "api/" + url + "/" + `?text=${text}`;
    console.log(url, text)
    debugger
    return dispatch => {
        axios
            .get(url)
            .then(res => {
                debugger
                dispatch(responseSuccess(res));
            })
            .catch(err => {
                const error = "Error 404 Page Not Found"
                dispatch(responseFailed(error));
            });
    };
};
