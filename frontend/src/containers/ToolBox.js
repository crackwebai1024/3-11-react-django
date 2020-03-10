import React from "react";
import "./Container.css";
import { connect } from "react-redux";
import { api_request } from "../store/actions/user";

class ToolBox extends React.Component {

    send_api_request = (e) => {
        var url = this.props.category;
        var text = document.getElementById("textcapture").value;
        this.props.api_request(url, text);
    }

    render() {
        var category = this.props.category;
        return (
            <div>
                <div style={{ height: 60 }}></div>
                <div className="txtinput">
                    <div style={{ height: 30 }}></div>
                    <div className="wrapperform">
                        <div><h5 className="title">{category}</h5></div>
                        <textarea className="form-control" id="textcapture"></textarea>
                        <button className="cusbtn" onClick={this.send_api_request}>Detect</button>
                    </div>
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