import React from "react";
import {
  Button,
  Form,
  Grid,
  Header,
  Message,
  Segment
} from "semantic-ui-react";
import { connect } from "react-redux";
import { NavLink, Redirect } from "react-router-dom";
import { authSignup } from "../store/actions/auth";

class RegistrationForm extends React.Component {
  state = {
    username: "",
    firstname: "",
    lastname: "",
    email: "",
    password1: "",
    password2: "",
    pwd1error: "",
    pwd2error: "",
  };

  handleSubmit = e => {
    e.preventDefault();
    const { username, firstname, lastname, email, password1, password2 } = this.state;
    this.props.signup(username, email, password1, password2);
    if (password1.length < 8) {
      this.setState({ pwd1error: "password must be more than 8 letters", pwd2error: "" })
    }
    if (password1.length >= 8 && password1 != password2) {
      this.setState({ pwd2error: "Your password are not confirmed correctly", pwd1error: "" })
    }
    if (password1.length >= 8 && password1 == password2) {
      this.setState({ pwd2error: "", pwd1error: "" })
    }
  };

  handleChange = e => {
    this.setState({ [e.target.name]: e.target.value });
    this.setState({ username: this.state.firstname + this.state.lastname });
  };

  render() {
    const { username, firstname, lastname, email, password1, password2 } = this.state;
    const { error, loading, token } = this.props;
    if (token) {
      return <Redirect to="/login" />;
    }
    return (
      <Grid
        textAlign="center"
        style={{ height: "100vh" }}
        verticalAlign="middle"
      >
        <Grid.Column style={{ maxWidth: 450 }}>
          <Header as="h2" color="teal" textAlign="center">
            Signup to your account
          </Header>
          <p className="error">{error}</p>

          <React.Fragment>
            <Form size="large" onSubmit={this.handleSubmit}>
              <Segment stacked>
                <Form.Input
                  onChange={this.handleChange}
                  value={firstname}
                  name="firstname"
                  fluid
                  icon="user"
                  iconPosition="right"
                  placeholder="Firstname"
                  required
                />
                <Form.Input
                  onChange={this.handleChange}
                  value={lastname}
                  name="lastname"
                  fluid
                  icon="user"
                  iconPosition="right"
                  placeholder="Lastname"
                  required
                />
                <Form.Input
                  onChange={this.handleChange}
                  value={email}
                  type="email"
                  name="email"
                  fluid
                  icon="mail"
                  iconPosition="right"
                  placeholder="E-mail address"
                  required
                />
                <Form.Input
                  onChange={this.handleChange}
                  fluid
                  value={password1}
                  name="password1"
                  icon="lock"
                  iconPosition="right"
                  placeholder="Password"
                  type="password"
                  required
                />
                <p className="error">{this.state.pwd1error}</p>
                <Form.Input
                  onChange={this.handleChange}
                  fluid
                  value={password2}
                  name="password2"
                  icon="lock"
                  iconPosition="right"
                  placeholder="Confirm password"
                  type="password"
                  required
                />
                <p className="error">{this.state.pwd2error}</p>
                <Button
                  color="teal"
                  fluid
                  size="large"
                  loading={loading}
                  disabled={loading}
                >
                  Signup
                </Button>
              </Segment>
            </Form>
            <Message>
              Already have an account? <NavLink to="/login">Login</NavLink>
            </Message>
          </React.Fragment>
        </Grid.Column>
      </Grid>
    );
  }
}

const mapStateToProps = state => {
  return {
    loading: state.auth.loading,
    error: state.auth.error,
    token: state.auth.token
  };
};

const mapDispatchToProps = dispatch => {
  return {
    signup: (username, email, password1, password2) =>
      dispatch(authSignup(username, email, password1, password2))
  };
};

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(RegistrationForm);
