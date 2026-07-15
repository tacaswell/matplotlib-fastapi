/**
 * Stamp package.json with the version from mpl_fastapi/_version.py before
 * npm publish.  This keeps the npm package version in sync with the Python
 * package version (both derived from git tags via setuptools-scm).
 *
 * Usage: node scripts/stamp-version.mjs
 *        (called automatically by the "prepublishOnly" npm lifecycle hook)
 */

import fs from 'fs';
import { execSync } from 'child_process';

function resolveVersion() {
  try {
    const py = fs.readFileSync('mpl_fastapi/_version.py', 'utf-8');
    const match = py.match(/version\s*=\s*"(.+)"/);
    if (match) return match[1];
  } catch (_) {}

  // Fallback: git describe (e.g. if _version.py hasn't been generated yet)
  try {
    return execSync('git describe --tags --always', { encoding: 'utf-8' }).trim();
  } catch (_) {}

  throw new Error('Cannot determine version: no _version.py and no git tags');
}

const version = resolveVersion();
execSync(`npm version --no-git-tag-version "${version}"`, { stdio: 'inherit' });
console.log(`📦 package.json version stamped: ${version}`);
