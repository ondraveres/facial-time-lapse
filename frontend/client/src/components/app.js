import React from "react";
import axios from 'axios';
import FileUploadItem from "./fileUploadItem";
import EnterNameItem from "./enterNameItem";
import Item from "./item";
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
    items: [
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
          <FileUploadItem handleAdd={this.handleAdd} />
        </div>



        <div className="mx-auto container flex justify-evenly flex-wrap">
          <div className=" w-96 mx-auto inline-block">
            <div className="md:mt-0 md:col-span-2">
              <div className="shadow sm:rounded-md sm:overflow-hidden">
                <div className="px-4 py-4 bg-white space-y-6 sm:p-6">
                  {/* <h3 className="font-medium leading-tight text-3xl mt-0 mb-2 text-blue-600">First image</h3> */}
                  <h1 className="text-3xl font-medium text-gray-900 text-center">Result timelapse</h1>

                  <form className="col-span-6 sm:col-span-3">

                    <div className="py-3 text-left flex justify-center">
                      <button
                        type="submit"
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      >
                        Generate gif
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className=" h-10"></div >

      </div >
    );
  }
}
