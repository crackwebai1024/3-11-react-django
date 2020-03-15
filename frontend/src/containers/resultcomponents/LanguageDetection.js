import React from "react";
import { connect } from "react-redux";

class LanguageDetection extends React.Component {

    render() {
        var display = { rawtext: "" };
        var text = "";
        display = this.props.data;
        if (display.rawtext != undefined) {
            text = "This sentence is written in " + display.rawtext + ".";
        }

        return (
            <div>
                <div style={{ height: 30 }}></div>
                <div>
                    <div style={{ height: 30 }}></div>
                    <div>
                        <h1>{text}</h1>.
                    </div>
                </div>
            </div>
        )
    }
}

const mapStateToProps = state => {
    return {
        data: state.user.data,
        error: state.user.reserror
    };
};

export default connect(
    mapStateToProps, null
)(LanguageDetection);