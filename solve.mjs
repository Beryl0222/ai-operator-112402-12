import { pathToFileURL } from 'node:url';
import { createSession } from './src/session.mjs';

export function solve(options = {}) {
  const session = createSession(options);
  const run = session.runEvents(options.events);
  return { ...run, report: session.report() };
}

export { createSession };

const invoked = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invoked) {
  const result = solve();
  console.log(JSON.stringify(result, null, 2));
}
