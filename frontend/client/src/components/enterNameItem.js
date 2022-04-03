import React, { Component } from "react";
import axios from 'axios';

export default class EnterNameItem extends Component {

  componentDidMount() {
    this.setState({ status: 'init' })
  }

  state = {
    id: this.props.id,
    description: this.props.description,
  };

  styles = {
    margin: "2vmin",
    cursor: "pointer"

  };
  tryToGetURL = () => {
    if (this.state.img) {
      return URL.createObjectURL(this.state.img)
    }
    else
      return null
  }

  handleSubmit = (e) => {
    e.preventDefault();
    this.setState({ status: 'waiting' })
    const data = { celebrityName: e.target[0].value }
    axios.post(`http://halmos.felk.cvut.cz:5000/uploadNameAPI`, data).then(res => {
      // then print response status
      console.log(res)
      res.data.forEach((pathAndAge) => {
        const path = pathAndAge[0]
        const age = pathAndAge[1]
        this.props.handleAdd(path, age)
      })
      // this.setState({ img: res.data[0] })
      this.setState({ status: 'init' })
    }).catch((error) => {
      console.log('123', error)
      this.setState({ status: 'failed' })
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        this.setState({ errorData: JSON.stringify(error.response.data), errorStatus: JSON.stringify(error.response.status) })
        console.log(error.response.data);
        console.log(error.response.status);
      } else if (error.request) {
        // The request was made but no response was received
        // `error.request` is an instance of XMLHttpRequest in the browser and an instance of
        // http.ClientRequest in node.js
        this.setState({ errorData: 'Servers are currently offline', errorStatus: 'No response' })
        console.log(error.request);
      } else {
        // Something happened in setting up the request that triggered an Error
        console.log('Error', error.message);
      }
    })

  }


  render() {
    return (
      // <div style={this.styles}>
      //   <span id='todo-item' onClick={() => this.props.onDelete(this.props.id)} style={this.styles}>{this.state.description} </span>
      //   <span id='tick' onClick={() => this.props.onDelete(this.props.id)}>✔️</span>
      // </div>
      <>


        <div className=" w-96 mx-auto inline-block">
          <div className="md:mt-0 md:col-span-2">
            <div className="shadow sm:rounded-md sm:overflow-hidden">
              <div className="px-4 py-4 bg-white space-y-6 sm:p-6">
                {/* <h3 className="font-medium leading-tight text-3xl mt-0 mb-2 text-blue-600">First image</h3> */}


                <h1 className="text-3xl font-medium text-gray-900 text-center">Enter a name</h1>


                <div className={this.state.status === 'init' ? "" : "hidden"} >

                  <form onSubmit={this.handleSubmit} className="col-span-6 sm:col-span-3">
                    <label htmlFor="first-name" className="block text-lg font-medium text-gray-700">
                      Celebrity name
                    </label>
                    <input
                      type="text"
                      name="first-name"
                      id="first-name"
                      autoComplete="given-name"
                      className="border-2 border-dashed mt-1 outline-none active:border-solid hover:border-indigo-500 block w-full shadow-sm text-lg border-gray-300 rounded-md p-1"
                    />

                    <div className="py-3 text-left">
                      <button
                        type="submit"
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      >
                        Run
                      </button>
                    </div>
                  </form>
                </div>
                <div className={this.state.status === 'waiting' ? "" : "hidden"}>
                  <div className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </div>
                </div>
                <div className={this.state.status === 'done' ? "" : "hidden"} >
                  <img src={this.tryToGetURL()} />
                </div>
                <div className={this.state.status === 'failed' ? "" : "hidden"} >
                  <p>Error status: {this.state.errorStatus}<br />
                    Error data: {this.state.errorData} </p>
                </div>
              </div>
            </div>
          </div>
        </div >
      </>

    );
  }
}


