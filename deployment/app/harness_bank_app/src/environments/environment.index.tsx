/* eslint-disable */
let environment: { production: boolean; apiUrl: string };

if (process.env.NODE_ENV === "production") {
  environment = require("./environment.prod").environment;
} else {
  environment = require("./environment").environment;
}

export default environment;
