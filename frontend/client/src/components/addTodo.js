import React, { Component, useState } from "react";

class AddTodo extends Component {
  buttonStyle = {
    margin: "20px",
  };

  constructor(props) {
    super(props);
    this.state = {
      inputValue: "",
    };
  }

  handleKeyPress = (event) => {
    if (event.key === "Enter") {
      this.props.onAdd(this.state.inputValue);
      this.setState({ inputValue: "" });


    }
  };

  render() {
    return (
      <div>
        <h3 id="add-todo">Add new todo</h3>
        <input
          style={this.textStyle}
          type="text"
          onChange={(e) => this.setState({ inputValue: e.target.value })}
          value={this.state.inputValue}
          onKeyPress={this.handleKeyPress}
        ></input>

        <span id="add-button"
          style={this.buttonStyle}
          onClick={() => {
            this.props.onAdd(this.state.inputValue);
            this.setState({ inputValue: "" });
          }}

        >
          ➕
        </span>
      </div>
    );
  }
}

export default AddTodo;
