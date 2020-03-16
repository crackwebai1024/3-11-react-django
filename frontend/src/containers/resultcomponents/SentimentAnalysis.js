import React from "react";
import { connect } from "react-redux";
import Chart from "./Chart";

class SentimentAnalysis extends React.Component {

    render() {
        return (
            <div>
                <div style={{ height: 30 }}></div>
                <div>
                    <Chart />
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
)(SentimentAnalysis);