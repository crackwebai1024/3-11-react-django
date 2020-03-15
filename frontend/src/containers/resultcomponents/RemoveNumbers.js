import React from "react";
import { connect } from "react-redux";

class RemoveNumbers extends React.Component {

    render() {
        var lowertxt = ""
        if (this.props.data != "") {
            lowertxt = this.props.data["result"]
        }
        return (
            <div>
                <div style={{ height: 30 }}></div>
                <div>
                    <div>
                        {lowertxt}
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
)(RemoveNumbers);