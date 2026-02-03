import esbuild from 'esbuild';
import { readFileSync } from 'fs';

const isWatch = process.argv.includes('--watch');

// Read package.json for version
const pkg = JSON.parse(readFileSync('./package.json', 'utf-8'));

const buildOptions = {
  entryPoints: ['mpl_fastapi/static/js/src/index.ts'],
  bundle: true,
  outfile: 'mpl_fastapi/static/js/dist/component.js',
  format: 'iife',
  globalName: 'MPL',
  sourcemap: true,
  target: 'es2020',
  minify: !isWatch,
  banner: {
    js: `/* matplotlib-fastapi v${pkg.version} | BSD-3-Clause License */`
  },
  define: {
    'process.env.NODE_ENV': isWatch ? '"development"' : '"production"'
  },
  logLevel: 'info'
};

if (isWatch) {
  console.log('🔨 Building TypeScript in watch mode...');
  const context = await esbuild.context(buildOptions);
  await context.watch();
  console.log('👀 Watching for changes...');
  console.log('Press Ctrl+C to stop');
} else {
  console.log('🔨 Building TypeScript...');
  await esbuild.build(buildOptions);
  console.log('✅ Build complete!');
}
