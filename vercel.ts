import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, sep } from 'node:path';

// `vercel build` bundles this file into <root>/.vercel/ before running it, so the
// module directory is only the repo root when the file runs from source.
const root = existsSync(resolve(import.meta.dirname, 'vercel.ts'))
  ? import.meta.dirname
  : resolve(import.meta.dirname, '..');

const cachePath = resolve(root, 'vercel.generated.json');

const skip = new Set([
  '.git',
  '.venv',
  '.vercel',
  'node_modules',
  '__pycache__',
  'dist',
  '.reference',
]);

async function askTheModel() {
  const { ToolLoopAgent, isStepCount, tool } = await import('ai');
  const { LocalFileSystemDetector, resolveAllConfiguredServicesV2 } = await import(
    '@vercel/fs-detectors'
  );
  const { getTransformedRoutes } = await import('@vercel/routing-utils');
  const { Ajv } = await import('ajv');
  const { z } = await import('zod');

  const detector = new LocalFileSystemDetector(root);

  const schema = (await (
    await fetch('https://openapi.vercel.sh/vercel.json')
  ).json()) as Record<string, unknown>;
  // The published schema is draft-04, which ajv 8 only reads as draft-07 once the
  // declaration is gone.
  delete schema.$schema;
  const validate = new Ajv({ allErrors: true, strict: false }).compile(schema);

  const inProject = (path: string) => {
    const full = resolve(root, path);
    if (full !== root && !full.startsWith(root + sep)) {
      throw new Error(`${path} is outside the project`);
    }
    return full;
  };

  const agent = new ToolLoopAgent({
    model: 'openai/gpt-5.6-luna',
    reasoning: 'high',
    stopWhen: isStepCount(40),
    instructions: `You generate the Vercel configuration for the repository you are looking at.

This is a polyglot monorepo that deploys as a single Vercel project using services: one
Vercel project builds several independent units (a frontend, one or more backends) and a
top-level route table sends public traffic to them. Read the docs before you decide
anything — start with "services", "services/config-reference" and "services/routing".

Your job:
1. Read the service docs so you know the exact schema and defaults.
2. Explore the repository and find every deployable unit. Look at the manifests
   (package.json, pyproject.toml, and so on) to work out each one's framework, runtime
   and entrypoint. A manifest may declare several entrypoints — a server plus background
   workers, say — so check that what you configure keeps all of them, rather than pinning
   one and silently dropping the rest.
3. Configure all of them. Do not stop at the obvious one, and do not invent services for
   directories that are only tests, fixtures or tooling.
4. Add the top-level rewrites that expose them. Rewrites are matched in order, so the
   catch-all must come last.

Only set a field when it differs from what Vercel detects on its own, and use whatever
other top-level configuration the repository turns out to need.

Run checkConfig on what you have before you answer, and fix anything it reports.

Answer with the finished configuration as a single JSON object — what the project's
vercel.json would contain — and nothing else.`,
    tools: {
      listFiles: tool({
        description: 'List a directory in the project. Directory names end with "/".',
        inputSchema: z.object({
          path: z.string().describe('Project-relative directory, "." for the repo root'),
        }),
        execute: ({ path }) =>
          readdirSync(inProject(path), { withFileTypes: true })
            .filter((entry) => !skip.has(entry.name))
            .map((entry) => (entry.isDirectory() ? `${entry.name}/` : entry.name)),
      }),
      readFile: tool({
        description: 'Read a text file in the project.',
        inputSchema: z.object({ path: z.string().describe('Project-relative file path') }),
        execute: ({ path }) => readFileSync(inProject(path), 'utf8').slice(0, 20_000),
      }),
      checkConfig: tool({
        description:
          'Check a candidate configuration against the published vercel.json schema and the route table rules the platform applies. Use it before you answer, and again after any change.',
        inputSchema: z.object({ config: z.string().describe('The candidate configuration, as JSON') }),
        execute: async ({ config }) => {
          let candidate;
          try {
            candidate = JSON.parse(config);
          } catch (error) {
            return `Not JSON: ${(error as Error).message}`;
          }

          const problems = validate(candidate)
            ? []
            : (validate.errors ?? []).map((e) => `${e.instancePath || '/'} ${e.message}`);

          // Resolves each service against the real filesystem, the same way the
          // platform does, so a service that cannot actually build says so here.
          const { errors } = await resolveAllConfiguredServicesV2(
            candidate.services ?? {},
            detector,
          );
          problems.push(...errors.map((e) => e.message));

          const { error } = getTransformedRoutes(candidate);
          if (error) problems.push(`${error.code}: ${error.message}`);

          for (const [index, rewrite] of (candidate.rewrites ?? []).entries()) {
            const name = rewrite?.destination?.service;
            if (name && !(name in (candidate.services ?? {}))) {
              problems.push(`rewrites[${index}] targets "${name}", which is not in services`);
            }
          }

          return problems.length ? problems : 'Valid.';
        },
      }),
      readVercelDocs: tool({
        description:
          'Read a page of the Vercel docs as markdown, e.g. "services/config-reference" for https://vercel.com/docs/services/config-reference.',
        inputSchema: z.object({ path: z.string().describe('Docs path without /docs/ or an extension') }),
        execute: async ({ path }) => {
          const response = await fetch(`https://vercel.com/docs/${path}.md`);
          return response.ok ? await response.text() : `${response.status} ${response.statusText}`;
        },
      }),
    },
    onToolExecutionStart: ({ toolCall }) =>
      console.error(`vercel.ts: ${toolCall.toolName} ${JSON.stringify(toolCall.input)}`),
  });

  const { text } = await agent.generate({ prompt: 'Configure this repository.' });
  const config = JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1));

  writeFileSync(cachePath, `${JSON.stringify(config, null, 2)}\n`);
  return config;
}

// The Vercel CLI kills the config loader after 10 seconds, which no agent will beat, so
// the generated config is committed and reused. `pnpm config:refresh` re-runs the model.
export default existsSync(cachePath) && !process.env.VERCEL_CONFIG_REFRESH
  ? JSON.parse(readFileSync(cachePath, 'utf8'))
  : await askTheModel();
