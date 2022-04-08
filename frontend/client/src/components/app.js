import React from "react";
import axios from 'axios';
import FileUploadItem from "./fileUploadItem";
import EnterNameItem from "./enterNameItem";
import Item from "./item";
import Result from "./result";
import Select from "./selectMenu";
import AddTodo from "./addTodo";


export default class extends React.Component {


  // componentDidMount() {
  //   fetch("http://localhost:3000/api/todos")
  //     .then((res) => res.json())
  //     .then((result) => {
  //       this.setState({
  //         items: result,
  //       });
  //     });
  // }

  state = {
    items: [{ id: 0, path: 'psp_example.png', age: 24 }
    ],
    length: 0
  };
  handleDelete = (todoId) => {
    const items = this.state.items.filter((todo) => todo.id !== todoId);
    this.setState({ items });

    fetch("http://localhost:3000/api/todos", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(items),
    });
  };

  handleAgeChange = (itemId, newAge) => {
    const items = this.state.items
    const itemIndex = items.findIndex((item => item.id == itemId));

    items[itemIndex].age = newAge;

    const orderedItems = items.sort((a, b) => {
      return a.age - b.age;
    });

    this.setState({ items: orderedItems });
  }

  handleAdd = (path, age) => {
    const items = this.state.items.concat({
      id: this.state.length + 1,
      path: path,
      age: age,
    });
    this.setState({
      length: this.state.length + 1,
    })
    const orderedItems = items.sort((a, b) => {
      return a.age - b.age;
    });

    this.setState({ items: orderedItems });
  };

  render() {
    return (

      <div>
        <div className="max-w-3xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-7xl my-3 font-bold text-gray-900 text-center">Facial timelapse generator</h1>
          <h3 className="text-3xl font-medium text-gray-400 text-center">Enter a name or upload your pictures</h3>
        </div>
        {/* <h1 className="text-3xl  underline">Facial time lapse video</h1> */}
        {/* <input type="file" onChange={this.uploadFile} /> */}

        <div className="mx-auto container flex justify-evenly align-middle">
          <EnterNameItem handleAdd={this.handleAdd} />
          <FileUploadItem handleAdd={this.handleAdd} />
          {/* <div className="flex align-middle justify-center flex-col mx-6"><h1 className="text-3xl font-medium text-gray-900 text-center">or</h1></div> */}
        </div>
        <div className="mx-auto container flex justify-evenly flex-wrap">
          {this.state.items.map((item) => (
            <Item
              key={item.id}
              id={item.id}
              path={item.path}
              age={item.age}
              onDelete={this.handleDelete}
              handleAgeChange={this.handleAgeChange}
            />
          ))}
        </div>


        <div className=" h-10"></div >
        <div className="mx-auto container flex justify-evenly flex-wrap">
          <Result items={this.state.items} />
        </div>

        <div className=" h-10"></div >


      </div >
    );
  }
}
