import React from "react";
import { connect } from "react-redux";

class RemoveStopwords extends React.Component {

    render() {
        var data = [];
        var stopwords = "";
        data = this.props.data["result"]
        if (data != []) {
            for (var x in data) {
                stopwords += " '" + data[x] + "', "
            }
        }

        return (
            <div>
                <div style={{ height: 30 }}></div>
                <div>
                    <div>
                        {stopwords}
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
)(RemoveStopwords);