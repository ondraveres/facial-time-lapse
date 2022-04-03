const path = require("path");
const express = require("express");
const bodyParser = require("body-parser");
const fs = require("fs");

const multer = require('multer');
const storage = multer.memoryStorage()

const upload = multer({ storage: storage })
const server = express();
const port = process.env.PORT || 3000;
const exists = require("util").promisify(require("fs").exists);

server.use('/storage', express.static(path.join(__dirname, "../../storage")))
server.use(bodyParser.urlencoded({ extended: false }));

server.use(bodyParser.json());

console.log(`NODE_ENV=${process.env.NODE_ENV}`);
state = [
  { id: 1, description: "Walk the dog" },
  { id: 2, description: "Do homework" },
  { id: 3, description: "Make lunch" },
  { id: 4, description: "Make lunch" },
];
server.get("/api/todos", (req, res) => {
  res.send(state);
});

server.post("/api/todos", (req, res) => {
  state = req.body;
  console.log(state);
});

const staticRoot =
  process.env.STATIC_ROOT || path.join(__dirname, "../client/dist");
if (!process.env.NODE_ENV) {
  const webpack = require("webpack");
  const config = require("../webpack.config.dev.js");
  const compiler = webpack(config);
  const webpackDevMiddleware = require("webpack-dev-middleware")(compiler, {
    publicPath: config.output.publicPath,
  });

  const webpackHotMiddleware = require("webpack-hot-middleware")(compiler);
  server.use(webpackDevMiddleware);
  server.use(webpackHotMiddleware);
}

exists(staticRoot).then((exists) => {
  if (exists) {
    console.log(`static assets directory located at ${staticRoot}`);
    server.use(express.static(staticRoot));
  }
});

server.listen(port, () => console.log(`node server running on ${port}`));

server.post('/uploadFileAPI', upload.single('file'), (req, res, next) => {
  console.log(req.file.buffer)

  // const file = req.file;
  // console.log('file name is', file);
  // if (!file) {
  //   const error = new Error('No File')
  //   error.httpStatusCode = 400
  //   return next(error)
  // }
  // //fs.writeFileSync(file,)
  // res.send(file);
})
