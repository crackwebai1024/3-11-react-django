import React from "react";
import { connect } from "react-redux";

class EntityExtraction extends React.Component {

    render() {
        var rawtext = "";
        var htmltxt = "";
        var header = "";
        if (this.props.data != "") {
            htmltxt = this.props.data["result"]
            document.getElementById("resulttext").innerHTML = htmltxt;
            rawtext = this.props.data["rawtext"]
            header = "Original Data"
        }
        return (
            <div>
                <div style={{ height: 30 }}></div>
                <div>
                    <div>
                        <h4>{header}</h4>
                    </div>
                    <div>{rawtext}</div>
                    <div id="resulttext"></div>
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
)(EntityExtraction);