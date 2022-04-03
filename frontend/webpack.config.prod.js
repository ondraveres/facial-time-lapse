const HtmlWebpackPlugin = require('html-webpack-plugin')
var WebpackMd5Hash = require('webpack-md5-hash');
const UglifyJsPlugin = require("uglifyjs-webpack-plugin");

let config;
try {
  config = require('./config/build.json')
  config.cdnPrefix = config.cdnPrefix.indexOf('$') === 0 ?
    process.env[config.cdnPrefix.substring(1)] :
    config.cdnPrefix || "";
} catch (err) {
  config = { cdnPrefix: "" };
}

const publicPath =  config.cdnPrefix || '';

module.exports = {
  mode: 'production',
  entry: { index: ['@babel/polyfill', './client/src/index.js'] },
  output: {
      path: `${__dirname}/client/dist`,
      filename: '[name].[chunkhash].js',
      publicPath: '/'
  },
  module: {
      rules: [
          {
              test: /\.js$/,
              exclude: /node_modules/,
              use: {
                  loader: 'babel-loader',
                  options: {
                      presets: [
                          '@babel/preset-env',
                          '@babel/preset-react'
                      ]
                  }
              }
          },
          {
              test: /\.css$/,
              use: [ 'style-loader', 'css-loader' ]
          }
      ]
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: 'client/src/index.html',
      publicPath
    }),
    new WebpackMd5Hash()
  ],
  optimization: {
      minimizer: [new UglifyJsPlugin({ include: /\.js$/ })]
  }
}
