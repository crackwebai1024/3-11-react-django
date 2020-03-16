import React, { Component } from 'react';
import { Canvas } from 'react-canvas-js';
import { connect } from 'react-redux'

class Chart extends Component {
    state = {
        data: ""
    }
    render() {
        var data = this.props.data
        if (data != this.state.data) {
            this.setState({ data: data })
            window.draw(data)
        }
        return (
            <div>
                <div id="chartContainer"></div>
                <div onLoad={this.draw}></div>
            </div>
        );
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
)(Chart);