import React from "react";
import { connect } from "react-redux";

class His extends React.Component {

    render() {
        const items = [];
        var data = this.props.hisdata;
        if (data != "") {
            var action = data["action"];
            var text = data["text"];
            var time = data["time"];
            var lent = action.length;
            for (var i = 0; i < lent; i++) {
                items.push(<tr>
                    <th scope="row">{i + 1}</th>
                    <td>{action[i]}</td>
                    <td className="textable">{text[i]}</td>
                    <td>{time[i]}</td>
                </tr>)
            }
        }
        return (
            <div className="hisdiv">
                <div style={{ height: 30 }}></div>
                <div>
                    <div>
                        <h3 className="histitle">History</h3>
                    </div>
                    <div>
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th scope="col">#</th>
                                    <th scope="col">action</th>
                                    <th scope="col" className="textable">text</th>
                                    <th scope="col">time</th>
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
        hisdata: state.user.hisdata,
        error: state.user.reserror
    };
};

export default connect(
    mapStateToProps, null
)(His);