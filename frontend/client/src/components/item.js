import React, { Component } from "react";
import axios from 'axios';

export default class Item extends Component {

  render() {
    return (

      <>
        <div className=" w-96 mx-auto inline-block my-4">
          <div className="md:mt-0 md:col-span-2">
            <div className="shadow sm:rounded-md sm:overflow-hidden">
              <div className="px-4 py-4 bg-white space-y-6 sm:p-6">

                {/* <h1 className="text-3xl font-medium text-gray-900 text-center">{this.props.id}</h1> */}



                <div className="justify-around flex">
                  <div className="mx-2">
                    <span className="inline-block h-32 w-32 overflow-hidden bg-gray-100">
                      {/* <svg className="h-full w-full text-gray-300" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M24 20.993V24H0v-2.996A14.977 14.977 0 0112.004 15c4.904 0 9.26 2.354 11.996 5.993zM16.002 8.999a4 4 0 11-8 0 4 4 0 018 0z" />
                      </svg> */}
                      <img src={'http://halmos.felk.cvut.cz:5000/storage/' + this.props.path} />
                    </span>
                  </div>
                  <div className="mx-2">
                    <label htmlFor="age" className="block text-lg font-medium text-gray-700">
                      Age
                    </label>
                    <input
                      type="text"
                      name="first-name"
                      id="age"
                      onChange={(e) => this.props.handleAgeChange(this.props.id, e.target.value)}
                      value={this.props.age}
                      className="border-2 border-dashed mt-1 outline-none active:border-solid hover:border-indigo-500 block w-full shadow-sm text-lg border-gray-300 rounded-md p-1"
                    />
                    <div className="py-3 text-left">
                      <button className="inline-flex justify-center py-2 px-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                        onClick={() => this.props.onDelete(this.props.id)}>
                        Remove
                      </button>
                    </div>

                  </div>
                </div>


              </div>
            </div>
          </div>
        </div >
      </>

    );
  }
}


