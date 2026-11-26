// Node 22 can require this ESM graph synchronously because it contains no
// top-level await.  This file is only a convenience for CJS-based probes.
module.exports = require('./index.mjs');
