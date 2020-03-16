import React from "react";
import "./Container.css";
import { connect } from "react-redux";
import { api_request } from "../store/actions/user";
import Result from "./Result";

class ToolBox extends React.Component {
    state = {
        idname: "textcapture",
        btncaption: "Search",
    }
    send_api_request = (e) => {
        var url = this.props.category;
        var text = document.getElementById(this.state.idname).value;
        this.props.api_request(url, text);
    }

    clear = () => {
        var textid = this.state.idname
        document.getElementById(textid).value = "";
    }
    render() {

        var category = this.props.category;

        if (category == "Article Recommender" && this.state.idname != "articlecapture") {
            this.setState({ idname: "articlecapture", btncaption: "Search" });
        }

        if (category != "Article Recommender" && this.state.idname == "articlecapture") {
            this.setState({ idname: "textcapture", btncaption: "Detect" });
        }

        return (
            <div>
                <div style={{ height: 40 }}></div>
                <div className="txtinput">
                    <div style={{ height: 30 }}></div>
                    <div className="wrapperform">
                        <div><h5 className="title">{category}</h5></div>
                        <textarea className="form-control" id={this.state.idname}></textarea>
                        <button className="cusbtn" onClick={this.send_api_request}>{this.state.btncaption}</button>
                        <button className="clrbtn" onClick={this.clear}>Clear</button>
                    </div>
                </div>
                <div>
                    <Result />
                </div>
            </div>
        )
    }
}

const mapStateToProps = state => {
    return {
        category: state.user.category
    };
};

const mapDispatchToProps = dispatch => {
    return {
        api_request: (url, text) => dispatch(api_request(url, text))
    };
};

export default connect(
    mapStateToProps, mapDispatchToProps
)(ToolBox);