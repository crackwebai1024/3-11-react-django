import React from "react";
import { connect } from "react-redux";

class ArticleExtraction extends React.Component {

    render() {
        return (
            <div>
                <div style={{ height: 30 }}></div>
                <div>
                    <div style={{ height: 30 }}></div>
                    <div>
                        <div><h5 className="title">Result</h5></div>
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
)(ArticleExtraction);