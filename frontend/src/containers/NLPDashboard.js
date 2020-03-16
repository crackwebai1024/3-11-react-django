import React, { Component } from "react";
import "./Container.css";
import { connect } from "react-redux";
import Category from "./Category.js";
import ToolBox from "./ToolBox.js";
import { user_history } from "../store/actions/user";
import His from "./His";

class NlpDashboard extends Component {

    seehistory = () => {
        this.props.seehistory();
    }


    render() {
        const { title } = this.props.title;
        var hisclick = this.props.click;

        if (!hisclick) {
            debugger
            var item = <ToolBox />
        }
        else {
            debugger
            var item = <His />
        }

        return (
            <div>
                <div style={{ height: 40 }}></div>
                <div className="row wrapper">
                    <div className="sidebar col-md-2">
                        <div className="imground">
                            <img src="embecia.png" className="emimage" />
                        </div>
                        <Category />
                    </div>
                    <div className="bodybar col-md-10">
                        <div className="bodymargin row">
                            <div className="col-md-3">
                                <h4 className="cattitle">{title}</h4>
                            </div>
                            <div className="col-md-8"></div>
                            <div className="col-md-1">
                                <button className="btn btn-primary right" onClick={this.seehistory}>See history</button>
                            </div>
                        </div>
                        {item}
                    </div>
                </div>
            </div>
        );
    }
}

const mapStateToProps = state => {
    return {
        title: state.user.title,
        click: state.user.clickhisbtn
    };
};

const mapDispatchToProps = dispatch => {
    return {
        seehistory: () => dispatch(user_history()),
    };
};

export default connect(
    mapStateToProps, mapDispatchToProps
)(NlpDashboard);