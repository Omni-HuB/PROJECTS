
// Creating Our First Simple NodeJs Server
// To create a server in Node.js, we need to use the http module. The http module provides a way of working with http requests and responses.

const http = require('http');

const server = http.createServer((req, res) => {
//   res.end('Hello World');
  console.log(req);
});

const PORT = 3001;
server.listen(PORT, () => {
  console.log('Server is running at','http//:localhost:${PORT})');
});

// To run the server, we need to execute the following command in the terminal:
// node firstServer.js  // Server is running at http//:localhost:3001