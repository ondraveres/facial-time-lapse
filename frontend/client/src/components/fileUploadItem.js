import React, { Component } from "react";
import axios from 'axios';

export default class FileUploadItem extends Component {

  componentDidMount() {
    this.setState({
      status: 'init',
      dragging: false,
      count: 0
    })
  }
  uploadFile = (event, myData = undefined) => {
    console.log('trying to upload file')
    this.setState({ status: 'waiting' })
    const data = new FormData();
    let files
    if (myData) {
      files = myData
    }
    else {
      files = event.target.files
    }
    console.log(files)
    for (const property in files) {
      if (property === 'length') {
        continue
      }
      data.append('image', files[property]);
      console.log('image')
    }

    axios.post(`http://halmos.felk.cvut.cz:5000/uploadFileAPI`, data)
      .then(res => { // then print response status
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

  state = {
    id: this.props.id,
    description: this.props.description,
  };

  tryToGetURL = () => {
    if (this.state.img) {
      return 'storage/' + this.state.img
    }
    else
      return null
  }

  handleDragEnter = e => {
    e.preventDefault();
    this.setState(prevState => {
      return {
        count: prevState.count + 1,
        dragging: true
      }
    })
  };

  handleDragLeave = e => {
    e.preventDefault();
    this.setState(prevState => {
      if (prevState.count === 1) {
        return {
          count: prevState.count - 1,
          dragging: false
        }
      }
      else {
        return {
          count: prevState.count - 1,
        }
      }

    })
  };

  handleDragOver = e => {
    e.preventDefault();
  };

  handleDrop = e => {
    e.preventDefault();
    this.myFileUpload.files = e.dataTransfer.files
    console.log(e.dataTransfer.files)
    this.uploadFile(e, e.dataTransfer.files)
  };

  render() {
    return (
      <>
        <div className=" w-96 mx-auto inline-block my-4">
          <div className="md:mt-0 md:col-span-2">
            <form action="#" method="POST">
              <div className="shadow sm:rounded-md sm:overflow-hidden">
                <div className="px-4 py-4 bg-white space-y-6 sm:p-6">
                  {/* <h3 className="font-medium leading-tight text-3xl mt-0 mb-2 text-blue-600">First image</h3> */}


                  {/* <h1 className="text-3xl font-medium text-gray-900 text-center">Upload images</h1> */}


                  {/* add flex */}
                  {/* <div className="justify-around hidden">
                    <div className="mx-2">
                      <label className="block text-sm font-medium text-gray-700">Photo</label>
                      <span className="inline-block h-32 w-32 overflow-hidden bg-gray-100">
                        <svg className="h-full w-full text-gray-300" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M24 20.993V24H0v-2.996A14.977 14.977 0 0112.004 15c4.904 0 9.26 2.354 11.996 5.993zM16.002 8.999a4 4 0 11-8 0 4 4 0 018 0z" />
                        </svg>
                      </span>
                    </div>
                    <div className="mx-2">
                      <label className="block text-sm font-medium text-gray-700">Inversion</label>
                      <span className="inline-block h-32 w-32 overflow-hidden bg-gray-100">
                        <svg className="h-full w-full text-gray-300" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M24 20.993V24H0v-2.996A14.977 14.977 0 0112.004 15c4.904 0 9.26 2.354 11.996 5.993zM16.002 8.999a4 4 0 11-8 0 4 4 0 018 0z" />
                        </svg>
                      </span>
                    </div>
                  </div> */}

                  <div className={this.state.status === 'init' ? "" : "hidden"} >
                    <div className={this.state.dragging ?
                      "mt-1 flex justify-center px-6 pt-4 pb-2 border-2 border-dashed rounded-md border-indigo-500"
                      :
                      "mt-1 flex justify-center px-6 pt-4 pb-2 border-2 border-gray-300 border-dashed rounded-md "}
                      // onDragOver={(event) => { event.preventDefault }}
                      // onDrop={(event) => { this.myFileUpload.files = event.dataTransfer.files; }}
                      // onDragEnter={(event) => { event.preventDefault() }}
                      onDrop={e => this.handleDrop(e)}
                      onDragOver={e => this.handleDragOver(e)}
                      onDragEnter={e => this.handleDragEnter(e)}
                      onDragLeave={e => this.handleDragLeave(e)}>
                      <div className="space-y-1 text-center">
                        <svg
                          className="mx-auto h-12 w-12 text-gray-400"
                          stroke="currentColor"
                          fill="none"
                          viewBox="0 0 48 48"
                          aria-hidden="true"
                        >
                          <path
                            d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                            strokeWidth={2}
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                        <div className="flex text-sm text-gray-600">
                          <label
                            htmlFor="file-upload"
                            className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500"
                          >
                            <span>Upload a file</span>
                            <input id="file-upload" name="file-upload" ref={ref => this.myFileUpload = ref} type="file" onChange={this.uploadFile} className="sr-only" multiple accept="image/png, image/jpeg" />
                          </label>
                          <p className="pl-1">or drag and drop</p>
                        </div>
                        <p className="text-xs text-gray-500">PNG or JPG</p>
                      </div>
                    </div>
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
            </form>
          </div>
        </div >
      </>

    );
  }
}


