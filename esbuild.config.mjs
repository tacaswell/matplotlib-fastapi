import esbuild from 'esbuild';
import fs from 'fs';
import { execSync } from 'child_process';

// Resolve version from (in priority order):
//   1. mpl_fastapi/_version.py  — present after `pip install` / Python build
//   2. git describe --tags --always  — present in a git checkout
//   3. fallback constant
function resolveVersion() {
  // 1. Read from the Python-generated _version.py (same source of truth as Python)
  try {
    const pyVersion = fs.readFileSync('mpl_fastapi/_version.py', 'utf-8');
    const match = pyVersion.match(/version\s*=\s*"(.+)"/);
    if (match) return match[1];
  } catch (_) {}

  // 2. Ask git
  try {
    return execSync('git describe --tags --always', { encoding: 'utf-8' }).trim();
  } catch (_) {}

  // 3. Give up
  console.warn('⚠️  Could not determine version; using fallback 0.0.0-dev');
  return '0.0.0-dev';
}

// Always regenerate so the version stays in sync with the current build.
const version = resolveVersion();
const versionFilePath = 'mpl_fastapi/static/js/src/_version.ts';
fs.writeFileSync(versionFilePath, `export const VERSION = "${version}";\n`);
console.log(`📝 _version.ts → ${version}`);

const isWatch = process.argv.includes('--watch');

const banner = {
  js: `/* matplotlib-fastapi | BSD-3-Clause License */`,
};

const commonOptions = {
  entryPoints: ['mpl_fastapi/static/js/src/index.ts'],
  bundle: true,
  sourcemap: true,
  target: 'es2020',
  banner,
  define: {
    'process.env.NODE_ENV': isWatch ? '"development"' : '"production"',
  },
  logLevel: 'info',
};

// Build for FastAPI static serving (IIFE bundle with global)
const iifeBuildOptions = {
  ...commonOptions,
  outfile: 'mpl_fastapi/static/js/dist/component.js',
  format: 'iife',
  globalName: 'MPL',
  minify: !isWatch,
};

// Build for FastAPI static serving (ESM bundle for <script type="module">)
const fastapiEsmBuildOptions = {
  ...commonOptions,
  outfile: 'mpl_fastapi/static/js/dist/component.esm.js',
  format: 'esm',
  minify: !isWatch,
};

// Build for npm package (ESM)
const esmBuildOptions = {
  ...commonOptions,
  outfile: 'dist/mpl-fastapi.js',
  format: 'esm',
  minify: !isWatch,
};

// Build for npm package (CJS)
const cjsBuildOptions = {
  ...commonOptions,
  outfile: 'dist/mpl-fastapi.cjs',
  format: 'cjs',
  minify: !isWatch,
};

if (isWatch) {
  console.log('🔨 Building TypeScript in watch mode...');
  // Watch all outputs so both FastAPI static and npm package stay fresh.
  // The React example imports from the npm ESM bundle (dist/mpl-fastapi.js),
  // so it must be rebuilt alongside the IIFE bundle.
  const contexts = await Promise.all([
    esbuild.context(iifeBuildOptions),
    esbuild.context(fastapiEsmBuildOptions),
    esbuild.context(esmBuildOptions),
    esbuild.context(cjsBuildOptions),
  ]);
  await Promise.all(contexts.map((ctx) => ctx.watch()));
  console.log('👀 Watching for changes...');
  console.log('Press Ctrl+C to stop');
} else {
  console.log('🔨 Building TypeScript (FastAPI static + npm package)...');
  await Promise.all([
    esbuild.build(iifeBuildOptions),
    esbuild.build(fastapiEsmBuildOptions),
    esbuild.build(esmBuildOptions),
    esbuild.build(cjsBuildOptions),
  ]);
  // Generate type declarations using tsc
  console.log('📝 Generating type declarations...');
  execSync('npx tsc --declaration --emitDeclarationOnly --outDir dist', {
    stdio: 'inherit',
  });
  // Create CJS type declaration
  const { readFileSync, writeFileSync } = await import('fs');
  const dtsContent = readFileSync('dist/index.d.ts', 'utf-8');
  writeFileSync('dist/index.d.cts', dtsContent);
  console.log('✅ Build complete!');
}
