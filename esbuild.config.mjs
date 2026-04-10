import esbuild from 'esbuild';
import { readFileSync } from 'fs';

const isWatch = process.argv.includes('--watch');

// Read package.json for version
const pkg = JSON.parse(readFileSync('./package.json', 'utf-8'));

const banner = {
  js: `/* matplotlib-fastapi v${pkg.version} | BSD-3-Clause License */`
};

const commonOptions = {
  entryPoints: ['mpl_fastapi/static/js/src/index.ts'],
  bundle: true,
  sourcemap: true,
  target: 'es2020',
  banner,
  define: {
    'process.env.NODE_ENV': isWatch ? '"development"' : '"production"'
  },
  logLevel: 'info'
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
  await Promise.all(contexts.map(ctx => ctx.watch()));
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
  const { execSync } = await import('child_process');
  execSync('npx tsc --declaration --emitDeclarationOnly --outDir dist', { stdio: 'inherit' });
  // Create CJS type declaration
  const dtsContent = readFileSync('dist/index.d.ts', 'utf-8');
  const { writeFileSync } = await import('fs');
  writeFileSync('dist/index.d.cts', dtsContent);
  console.log('✅ Build complete!');
}
