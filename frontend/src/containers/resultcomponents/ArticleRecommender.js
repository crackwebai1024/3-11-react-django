import React from "react";
import { connect } from "react-redux";

class ArticleRecommender extends React.Component {

    render() {
        var display = "";
        const items = [];
        var i = 0;
        if (this.props.data != "") {
            display = this.props.data.tables;
            for (var x of display) {
                i += 1;
                items.push(<tr>
                    <th scope="row">{i}</th>
                    <td>{x[0]}</td>
                    <td>{x[1]}</td>
                </tr>)
            }
        }

        if (this.props.error != "") {
            items.push(this.props.reserror)
            console.log(items)
        }
        return (
            <div>
                <div style={{ height: 30 }}></div>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th scope="col">#</th>
                            <th scope="col">Article</th>
                            <th scope="col">URL</th>
                        </tr>
                    </thead>
                    <tbody>
                        {items}
                    </tbody>
                </table>
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
)(ArticleRecommender);