
const { execSync } = require("child_process");
const { version } = require('../package.json');

const envValue = param =>
  param.indexOf('$') === 0 ? process.env[param.substring(1)] : param;

let config;
try {
  config = require('../config/build.json');
  config.dockerVersion = envValue(config.dockerVersion) || version;
  config.dockerRepo = envValue(config.dockerRepo) || "express-react-scaffold";
  config.cdnPrefix = envValue(config.cdnPrefix) || "";
} catch (err) {
  console.log("build.json is not defined, using defaults.")
  config = {
    dockerRepo: "express-react-scaffold",
    dockerVersion: version,
    cdnPrefix: ""
  };
}

const bundleAssets = config.cdnPrefix ? false :true;
const dockerTag = `${config.dockerRepo}:${config.dockerVersion}`;
execSync(`docker build -t ${dockerTag} --build-arg BUNDLE_ASSETS=${bundleAssets} .`, {stdio: 'inherit'});
