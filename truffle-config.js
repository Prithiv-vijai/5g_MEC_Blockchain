module.exports = {
  networks: {
    development: {
      host: "127.0.0.1", // Localhost (default: none)
      port: 7545,        // Ganache default port
      network_id: "5777",   // Match any network id
    },
  },
  // Set default mocha options here, use special reporters, etc.
  mocha: {
    // timeout: 100000
  },
  // Configure your compilers
  compilers: {
    solc: {
      version: "0.8.0", // Fetch exact version from solc-bin (default: truffle's version)
    },
  },
  // Truffle DB is currently disabled by default; enable it to save your migrations
  db: {
    enabled: true,
  },
};
