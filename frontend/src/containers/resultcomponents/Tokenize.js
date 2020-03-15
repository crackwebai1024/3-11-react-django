import React from "react";
import { connect } from "react-redux";

class Tokenize extends React.Component {

    render() {
        var data = [];
        var tokenwords = "";
        data = this.props.data["result"]
        debugger
        if (data != []) {
            for (var x in data) {
                tokenwords += " '" + data[x] + "', "
                console.log(tokenwords)
            }
        }

        return (
            <div>
                <div style={{ height: 30 }}></div>
                <div>
                    <div>
                        {tokenwords}
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
)(Tokenize);