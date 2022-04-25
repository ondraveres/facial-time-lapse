import React, { Component } from "react";
import axios from 'axios';

var ReactDOM = require('react-dom');
var GifPlayer = require('react-gif-player');
import 'tw-elements';

import Dropdown from "./dropdown";

export default class Result extends Component {



  state = {
    status: 'init',
    opacity: this.props.defaultOpacity,
    frames: this.props.defaultFrames,
    encoder: this.props.defaultEncoder,
    size: this.props.defaultSize,
    title: this.props.title,
  };

  tryToGetURL = () => {

    if (this.props.image) {
      return URL.createObjectURL(this.props.image)
    }

    else
      return null
  }

  handleSubmit = (e) => {
    e.preventDefault();
    this.setState({ status: 'waiting' })
    console.log('form data: ', e.target)
    const data = {
      encoder: this.state.encoder,
      size: this.state.size,
      frames: this.state.frames,
      opacity: this.state.opacity,
      items: this.props.items
    }
    const url = 'http://halmos.felk.cvut.cz:5000/generateGifAPI'
    axios.post(url, data, { responseType: 'blob' }).then(res => {
      // then print response status
      console.log(res)
      this.props.updateImages(res.data, this.props.v)
      this.setState({ img: res.data })
      this.setState({ status: 'done' })
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


        <div className="w-96 mx-auto inline-block my-4">
          <div className="md:mt-0 md:col-span-2">
            <div className="shadow sm:rounded-md">
              <div className="px-4 py-4 bg-white space-y-6 sm:p-6">
                {/* <h3 className="font-medium leading-tight text-3xl mt-0 mb-2 text-blue-600">First image</h3> */}


                <h1 className="text-3xl font-medium text-gray-900 text-center">{this.state.title}</h1>


                <div className={this.state.status === 'init' ? "my-0" : "hidden"}>

                  <form onSubmit={this.handleSubmit} className="">
                    <div className="accordion" id="accordionExample">

                      <div className="accordion-item bg-white border border-gray-200">
                        <h2 className="accordion-header mb-0" id="headingOne">
                          <button className="accordion-button collapsed relative flex items-center w-full py-2 px-3 text-base text-gray-800 text-left bg-white border-0 rounded-none transition  focus:outline-none"
                            type="button" data-bs-toggle="collapse"
                            data-bs-target={'#' + this.props.uniqueid}
                          >
                            Settings
                          </button>
                        </h2>

                        <div id={this.props.uniqueid} className="accordion-collapse border-0 collapse">
                          <div className="accordion-body py-1 px-2">


                            <div className="flex items-center justify-between">
                              <label htmlFor="country" className="block text-sm font-medium text-gray-700">
                                Encoder:
                              </label>
                              <select
                                id="encoder"
                                name="encoder"
                                onChange={(e) => {
                                  this.setState({
                                    encoder: event.target.value,
                                    title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`,
                                  })
                                }
                                }
                                value={this.state.encoder}
                                className="w-52 mt-1 block py-2 pl-2 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                              >
                                <option value="psp">StyleGAN2 pSp</option>
                                <option value="toonify">StyleGAN2 pSp Toonify</option>
                                <option value="restyle">StyleGAN3 ReStyle pSp</option>
                                <option value="pixel">None</option>
                              </select>
                            </div>

                            <div className="flex items-center justify-between">
                              <label htmlFor="country" className="block text-sm font-medium text-gray-700">
                                Output gif size:
                              </label>
                              <select
                                id="size"
                                name="size"
                                onChange={(e) => {
                                  this.setState({
                                    size: event.target.value,
                                    title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`
                                  })
                                }
                                }
                                value={this.state.size}
                                className="w-52 mt-1 block py-2 pl-2 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                              >
                                <option value="1024">1024x1024 px</option>
                                <option value="512">512x512 px</option>
                                <option value="256">256x256 px</option>
                              </select>
                            </div>



                            <div className="flex items-center justify-between">
                              <label htmlFor="first-name" className="block text-sm font-medium text-gray-700">
                                Frames between two images
                              </label>
                              <input
                                type="text"
                                name="first-name"
                                id="first-name"
                                autoComplete="given-name"
                                onChange={(e) => {
                                  const value = parseInt(event.target.value)
                                  if (value)
                                    this.setState({
                                      frames: value <= 60 ? value : 60,
                                      title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`,
                                    })
                                }
                                }
                                value={this.state.frames}
                                className="px-0 text-center w-12 mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                              />
                            </div>

                            <div className="flex items-center justify-between">
                              <label htmlFor="first-name" className="block text-sm font-medium text-gray-700">
                                Max blended image opacity (%)
                              </label>
                              <input
                                type="text"
                                name="first-name"
                                id="first-name"
                                autoComplete="given-name"
                                onChange={(e) => {
                                  const value = parseInt(event.target.value)
                                  if (!Number.isNaN(value))
                                    this.setState({
                                      opacity: value <= 100 ? value : 100,
                                      title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`,
                                    })
                                }
                                }
                                value={this.state.opacity}
                                className="px-0 text-center w-12 mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>


                    {/* <Dropdown /> */}

                    <div className="py-3 text-left flex justify-center">
                      <button
                        type="submit"
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 justify-center"
                      >
                        Generate gif
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

                  {/* <GifPlayer gif={this.tryToGetURL()} /> */}
                  <img src={this.tryToGetURL()} />


                  <form onSubmit={this.handleSubmit} className="col-span-6 sm:col-span-3">
                    <div className="accordion" id="accordionExample">

                      <div className="accordion-item bg-white border border-gray-200">
                        <h2 className="accordion-header mb-0" id="headingOne">
                          <button className="accordion-button collapsed relative flex items-center w-full py-2 px-3 text-base text-gray-800 text-left bg-white border-0 rounded-none transition  focus:outline-none"
                            type="button" data-bs-toggle="collapse"
                            data-bs-target={'#' + this.props.uniqueid}
                          >
                            Settings
    </button>
                        </h2>

                        <div id={this.props.uniqueid} className="accordion-collapse border-0 collapse">
                          <div className="accordion-body py-1 px-2">


                            <div className="flex items-center justify-between">
                              <label htmlFor="country" className="block text-sm font-medium text-gray-700">
                                Encoder:
        </label>
                              <select
                                id="encoder"
                                name="encoder"
                                onChange={(e) => {
                                  this.setState({
                                    encoder: event.target.value,
                                    title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`,
                                  })
                                }
                                }
                                value={this.state.encoder}
                                className="w-52 mt-1 block py-2 pl-2 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                              >
                                <option value="psp">StyleGAN2 pSp</option>
                                <option value="toonify">StyleGAN2 pSp Toonify</option>
                                <option value="restyle">StyleGAN3 ReStyle pSp</option>
                                <option value="pixel">None</option>
                              </select>
                            </div>

                            <div className="flex items-center justify-between">
                              <label htmlFor="country" className="block text-sm font-medium text-gray-700">
                                Output gif size:
        </label>
                              <select
                                id="size"
                                name="size"
                                onChange={(e) => {
                                  this.setState({
                                    size: event.target.value,
                                    title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`
                                  })
                                }
                                }
                                value={this.state.size}
                                className="w-52 mt-1 block py-2 pl-2 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                              >
                                <option value="1024">1024x1024 px</option>
                                <option value="512">512x512 px</option>
                                <option value="256">256x256 px</option>
                              </select>
                            </div>



                            <div className="flex items-center justify-between">
                              <label htmlFor="first-name" className="block text-sm font-medium text-gray-700">
                                Frames between two images
        </label>
                              <input
                                type="text"
                                name="first-name"
                                id="first-name"
                                autoComplete="given-name"
                                onChange={(e) => {
                                  const value = parseInt(event.target.value)
                                  if (value)
                                    this.setState({
                                      frames: value <= 60 ? value : 60,
                                      title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`,
                                    })
                                }
                                }
                                value={this.state.frames}
                                className="px-0 text-center w-12 mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                              />
                            </div>

                            <div className="flex items-center justify-between">
                              <label htmlFor="first-name" className="block text-sm font-medium text-gray-700">
                                Max blended image opacity (%)
        </label>
                              <input
                                type="text"
                                name="first-name"
                                id="first-name"
                                autoComplete="given-name"
                                onChange={(e) => {
                                  const value = parseInt(event.target.value)
                                  if (!Number.isNaN(value))
                                    this.setState({
                                      opacity: value <= 100 ? value : 100,
                                      title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`,
                                    })
                                }
                                }
                                value={this.state.opacity}
                                className="px-0 text-center w-12 mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="py-3 text-left flex justify-center">
                      <button
                        type="submit"
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 justify-center"
                      >
                        Regenerate gif
                      </button>
                    </div>
                  </form>
                </div>
                <div className={this.state.status === 'failed' ? "" : "hidden"} >
                  <p>Error status: {this.state.errorStatus}<br />
                    Error data: {this.state.errorData} </p>
                  <form onSubmit={this.handleSubmit} className="col-span-6 sm:col-span-3">
                    <div className="accordion" id="accordionExample">

                      <div className="accordion-item bg-white border border-gray-200">
                        <h2 className="accordion-header mb-0" id="headingOne">
                          <button className="accordion-button collapsed relative flex items-center w-full py-2 px-3 text-base text-gray-800 text-left bg-white border-0 rounded-none transition  focus:outline-none"
                            type="button" data-bs-toggle="collapse"
                            data-bs-target={'#' + this.props.uniqueid}
                          >
                            Settings
                          </button>
                        </h2>

                        <div id={this.props.uniqueid} className="accordion-collapse border-0 collapse">
                          <div className="accordion-body py-1 px-2">


                            <div className="flex items-center justify-between">
                              <label htmlFor="country" className="block text-sm font-medium text-gray-700">
                                Encoder:
                              </label>
                              <select
                                id="encoder"
                                name="encoder"
                                onChange={(e) => {
                                  this.setState({
                                    encoder: event.target.value,
                                    title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`,
                                  })
                                }
                                }
                                value={this.state.encoder}
                                className="w-52 mt-1 block py-2 pl-2 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                              >
                                <option value="psp">StyleGAN2 pSp</option>
                                <option value="toonify">StyleGAN2 pSp Toonify</option>
                                <option value="restyle">StyleGAN3 ReStyle pSp</option>
                                <option value="pixel">None</option>
                              </select>
                            </div>

                            <div className="flex items-center justify-between">
                              <label htmlFor="country" className="block text-sm font-medium text-gray-700">
                                Output gif size:
                              </label>
                              <select
                                id="size"
                                name="size"
                                onChange={(e) => {
                                  this.setState({
                                    size: event.target.value,
                                    title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`
                                  })
                                }
                                }
                                value={this.state.size}
                                className="w-52 mt-1 block py-2 pl-2 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                              >
                                <option value="1024">1024x1024 px</option>
                                <option value="512">512x512 px</option>
                                <option value="256">256x256 px</option>
                              </select>
                            </div>



                            <div className="flex items-center justify-between">
                              <label htmlFor="first-name" className="block text-sm font-medium text-gray-700">
                                Frames between two images
        </label>
                              <input
                                type="text"
                                name="first-name"
                                id="first-name"
                                autoComplete="given-name"
                                onChange={(e) => {
                                  const value = parseInt(event.target.value)
                                  if (value)
                                    this.setState({
                                      frames: value <= 60 ? value : 60,
                                      title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`,
                                    })
                                }
                                }
                                value={this.state.frames}
                                className="px-0 text-center w-12 mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                              />
                            </div>

                            <div className="flex items-center justify-between">
                              <label htmlFor="first-name" className="block text-sm font-medium text-gray-700">
                                Max blended image opacity (%)
        </label>
                              <input
                                type="text"
                                name="first-name"
                                id="first-name"
                                autoComplete="given-name"
                                onChange={(e) => {
                                  const value = parseInt(event.target.value)
                                  if (!Number.isNaN(value))
                                    this.setState({
                                      opacity: value <= 100 ? value : 100,
                                      title: `Custom settings ${(this.props.uniqueid).toUpperCase()}`,
                                    })
                                }
                                }
                                value={this.state.opacity}
                                className="px-0 text-center w-12 mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="py-3 text-left flex justify-center">
                      <button
                        type="submit"
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 justify-center"
                      >
                        Try again
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </div>
        </div >
      </>

    );
  }
}


