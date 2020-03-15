import React from "react";
import { connect } from "react-redux";

class TextSummarization extends React.Component {

    render() {
        const items = [];
        var data = this.props.data;
        if (data != "") {
            var summarizetext = data["textdata"][1];
            var senscore = data["textdata"][0];
            var i = 0;
            for (var x of senscore) {
                i += 1;
                items.push(<tr>
                    <th scope="row">{i}</th>
                    <td>{x[0]}</td>
                    <td>{x[1]}</td>
                </tr>)
            }
        }
        return (
            <div>
                <div style={{ height: 30 }}></div>
                <div>
                    <div>
                        Summarize Text
                    </div>
                    <div>
                        {summarizetext}
                    </div>
                    <div>
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th scope="col">#</th>
                                    <th scope="col">Score</th>
                                    <th scope="col">Sentence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {items}
                            </tbody>
                        </table>
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
)(TextSummarization);