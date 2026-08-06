// E2E image-latency test: how fast does seal get pictures on screen?
//
//   1. open seal in a fresh empty chat
//   2. send "draw N pictures of things you find interesting" (default N=5)
//   3. time each generated image from submit to the moment it paints
//   4. report time-to-first-image and time-to-all-N
//
// Timings come from an in-page MutationObserver that stamps every new
// <img alt="Generated image"> when it finishes loading, so they do not depend
// on this script's polling cadence. Tool-card appearances are stamped too,
// which separates "model asked for an image" from "image came back".
//
// Setup (once): pnpm install && pnpm run install-browser
// Run against a live server (default http://localhost:3000):
//   pnpm run test:images
//   N=3 HEADED=1 SEAL_URL=http://localhost:3000 node images.mjs
//
// Env: SEAL_URL, N, PROMPT, DEADLINE_MS, STALL_MS, HEADED, NO_COLOR.
// Exits 0 when all N images rendered, non-zero otherwise. Screenshot and a
// JSON summary land in /tmp/seal-e2e-images-*.

import { writeFileSync } from "node:fs";

import { chromium } from "playwright";

const SEAL_URL = process.env.SEAL_URL ?? "http://localhost:3000";
const HEADED = !!process.env.HEADED;
const N = Number(process.env.N ?? 5);
const PROMPT =
  process.env.PROMPT ?? `draw ${N} pictures of things you find interesting`;

// image generation is slow and bursty, so these are much looser than the
// tool-coverage test's timeouts.
const TIMEOUT_MS = {
  action: 10_000,
  navigation: 15_000,
  appReady: 15_000,
  emptyState: 15_000,
  promptVisible: 15_000,
  deadline: Number(process.env.DEADLINE_MS ?? 600_000),
  stalled: Number(process.env.STALL_MS ?? 120_000),
  tailAfterImages: 60_000,
  heartbeat: 2_000,
  poll: 250,
  afterApproval: 250,
  headedPause: 3_000,
};

if (!Number.isInteger(N) || N < 1) {
  console.error(`N must be a positive integer, got ${process.env.N}`);
  process.exit(2);
}

const colors =
  process.env.NO_COLOR || process.env.TERM === "dumb"
    ? { red: "", green: "", yellow: "", cyan: "", dim: "", bold: "", reset: "" }
    : {
        red: "\x1b[31m",
        green: "\x1b[32m",
        yellow: "\x1b[33m",
        cyan: "\x1b[36m",
        dim: "\x1b[2m",
        bold: "\x1b[1m",
        reset: "\x1b[0m",
      };

const color = (style, text) => `${colors[style]}${text}${colors.reset}`;
const failures = [];

const log = (msg) => console.log(`${color("dim", "-")} ${msg}`);
const pass = (msg) => console.log(`${color("green", "PASS")} ${msg}`);
const warn = (msg) => console.warn(`${color("yellow", "WARN")} ${msg}`);
const fail = (msg) => {
  failures.push(msg);
  console.error(`${color("red", "FAIL")} ${msg}`);
  process.exitCode = 1;
};

const secs = (ms) => (ms == null ? "n/a" : `${(ms / 1000).toFixed(2)}s`);

// Stamps generated images and generate_image tool cards as they show up. Runs
// on every document so it survives a reload mid-run. Timestamps are epoch ms
// (timeOrigin + performance.now()) so they compare directly against Date.now()
// taken on this side when the prompt is submitted.
function observer() {
  const marks = { images: [], cards: [] };
  window.__sealMarks = marks;
  const seen = new Set();
  const now = () => performance.timeOrigin + performance.now();

  const note = (img) => {
    // key on src, not element identity: React can remount an <img> for an
    // image that already painted, and that must not count twice.
    const src = img.getAttribute("src") ?? "";
    if (!src || seen.has(src)) return;
    seen.add(src);
    const stamp = () => marks.images.push({ t: now(), bytes: src.length });
    if (img.complete && img.naturalWidth > 0) stamp();
    else img.addEventListener("load", stamp, { once: true });
  };

  const scan = () => {
    const root = document.querySelector('[data-testid="chat-log"]');
    if (!root) return;
    for (const img of root.querySelectorAll('img[alt="Generated image"]'))
      note(img);
    const cards = root.querySelectorAll(
      '[data-testid="tool-card"][data-tool-name="generate_image"]',
    ).length;
    while (marks.cards.length < cards) marks.cards.push(now());
  };

  // `document`, not `document.documentElement`: init scripts run before the
  // document is parsed, so documentElement is still null here.
  new MutationObserver(scan).observe(document, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["src", "data-tool-state", "data-tool-name"],
  });
  scan();
}

const chatLog = (page) => page.getByTestId("chat-log");

// The sidebar lists sessions titled with LLM-generated text, so locators for
// app chrome must never search page-wide (substring matches go ambiguous).
const mainPane = (page) => page.getByRole("main");

function toolState(page, state) {
  return chatLog(page).locator(
    `[data-testid="tool-card"][data-tool-depth="0"][data-tool-state="${state}"]`,
  );
}

// True when the conversation ends with the agent's own answer rather than a
// tool card -- i.e. the turn really finished instead of dying after a tool.
function finalAnswerPresent(page) {
  return page.evaluate(() => {
    const root = document.querySelector('[data-testid="chat-log"]');
    if (!root) return false;
    const nodes = root.querySelectorAll(
      '[data-testid="message"][data-message-depth="0"],' +
        '[data-testid="tool-card"][data-tool-depth="0"]',
    );
    const last = nodes[nodes.length - 1];
    return (
      !!last &&
      last.getAttribute("data-testid") === "message" &&
      last.getAttribute("data-message-role") === "assistant" &&
      (last.textContent || "").trim().length > 0
    );
  });
}

async function snapshot(page) {
  const chat = chatLog(page);
  const [marks, approve, completed, errored, denied, streaming, answered] =
    await Promise.all([
      page.evaluate(() => window.__sealMarks ?? null),
      chat.getByRole("button", { name: /approve/i }).count(),
      toolState(page, "output-available").count(),
      toolState(page, "output-error").count(),
      toolState(page, "output-denied").count(),
      mainPane(page).getByRole("button", { name: "Stop", exact: true }).count(),
      finalAnswerPresent(page),
    ]);
  // without the observer every timing would silently read as zero, so treat a
  // missing one as fatal rather than as "no images yet".
  if (!marks) throw new Error("the in-page image observer is not installed");
  return {
    images: marks.images.map((m) => m.t).sort((a, b) => a - b),
    bytes: marks.images.map((m) => m.bytes),
    cards: marks.cards.sort((a, b) => a - b),
    approve,
    completed,
    errored,
    denied,
    streaming: streaming > 0,
    answered,
  };
}

const describe = (s) =>
  `images:${s.images.length}/${N} cards:${s.cards.length} ` +
  `done:${s.completed} err:${s.errored} denied:${s.denied} ` +
  `streaming:${s.streaming} answered:${s.answered}`;

const signature = (s) =>
  JSON.stringify([
    s.images.length,
    s.cards.length,
    s.approve,
    s.completed,
    s.errored,
    s.denied,
    s.streaming,
    s.answered,
  ]);

async function shot(page, label) {
  await page
    .screenshot({ path: `/tmp/seal-e2e-images-${label}.png`, fullPage: true })
    .catch(() => {});
}

// Polls until `done(s)` or a budget runs out. Approves any gated tool along the
// way so an unexpected bash call cannot wedge the run.
async function waitFor(page, { done, budgetMs, what }) {
  const startedAt = Date.now();
  let lastSignature = "";
  let lastProgressAt = startedAt;
  let lastLogAt = 0;
  let approvals = 0;

  while (true) {
    const s = await snapshot(page);
    const now = Date.now();
    const sig = signature(s);
    if (sig !== lastSignature) {
      lastSignature = sig;
      lastProgressAt = now;
    }
    if (now - lastLogAt > TIMEOUT_MS.heartbeat) {
      log(`...${what} (${describe(s)})`);
      lastLogAt = now;
    }
    if (done(s)) return { snapshot: s, approvals, timedOut: false };

    if (s.approve > 0) {
      await chatLog(page)
        .getByRole("button", { name: /approve/i })
        .first()
        .click({ timeout: TIMEOUT_MS.action })
        .catch(() => {});
      approvals++;
      log(`approved a gated tool execution (#${approvals})`);
      await page.waitForTimeout(TIMEOUT_MS.afterApproval);
      continue;
    }

    if (now - lastProgressAt > TIMEOUT_MS.stalled) {
      log(`no progress for ${secs(TIMEOUT_MS.stalled)} while ${what}`);
      return { snapshot: s, approvals, timedOut: true, reason: "stalled" };
    }
    if (now - startedAt > budgetMs) {
      log(`budget of ${secs(budgetMs)} exhausted while ${what}`);
      return { snapshot: s, approvals, timedOut: true, reason: "deadline" };
    }
    await page.waitForTimeout(TIMEOUT_MS.poll);
  }
}

function report(t0, s, tAnswer) {
  const rel = (t) => Math.round(t - t0);
  const images = s.images.map(rel);
  const cards = s.cards.map(rel);

  console.log(`\n${color("bold", "== timings ==")}`);
  console.log(`${color("dim", "prompt".padEnd(22))} ${PROMPT}`);
  console.log(
    `${color("dim", "first tool call".padEnd(22))} ${secs(cards[0])}` +
      `${color("dim", "  (generate_image card appeared)")}`,
  );
  console.log(
    `${color("cyan", "time to first image".padEnd(22))} ${secs(images[0])}`,
  );
  console.log(
    `${color("cyan", `time to all ${N} images`.padEnd(22))} ` +
      `${secs(images[N - 1])}`,
  );
  if (tAnswer != null) {
    console.log(
      `${color("dim", "time to final answer".padEnd(22))} ${secs(rel(tAnswer))}`,
    );
  }

  console.log(`\n${color("bold", "== per image ==")}`);
  images.forEach((t, i) => {
    const gap = i === 0 ? t : t - images[i - 1];
    const kb = Math.round((s.bytes[i] ?? 0) / 1024);
    console.log(
      `  #${String(i + 1).padStart(2)} at ${secs(t).padStart(8)} ` +
        `${color("dim", `(+${secs(gap)}, ~${kb}kB data url)`)}`,
    );
  });

  const summary = {
    url: SEAL_URL,
    prompt: PROMPT,
    n: N,
    rendered: images.length,
    firstToolCallMs: cards[0] ?? null,
    firstImageMs: images[0] ?? null,
    allImagesMs: images[N - 1] ?? null,
    finalAnswerMs: tAnswer == null ? null : rel(tAnswer),
    imageMs: images,
    toolCardMs: cards,
  };
  const path = "/tmp/seal-e2e-images-summary.json";
  writeFileSync(path, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(`\n${color("bold", "== json ==")} ${color("dim", path)}`);
  console.log(JSON.stringify(summary));
}

const browser = await chromium.launch({
  headless: !HEADED,
  timeout: TIMEOUT_MS.action,
});
const context = await browser.newContext({
  viewport: { width: 1280, height: 900 },
});
await context.addInitScript(observer);
const page = await context.newPage();
page.setDefaultTimeout(TIMEOUT_MS.action);
page.setDefaultNavigationTimeout(TIMEOUT_MS.navigation);

try {
  log(`opening ${SEAL_URL}`);
  await page.goto(SEAL_URL, {
    waitUntil: "domcontentloaded",
    timeout: TIMEOUT_MS.navigation,
  });
  const textarea = mainPane(page).getByPlaceholder("Ask me anything...");
  await textarea.waitFor({ state: "visible", timeout: TIMEOUT_MS.appReady });
  await chatLog(page)
    .getByText("Start a conversation")
    .waitFor({ state: "visible", timeout: TIMEOUT_MS.emptyState });
  log("app ready in a fresh empty chat");

  await textarea.fill(PROMPT);
  const submit = mainPane(page).getByRole("button", {
    name: "Submit",
    exact: true,
  });
  await submit.waitFor({ state: "visible" });

  // t0: everything is measured from the submit click.
  const t0 = Date.now();
  await submit.click();
  log(`sent prompt: "${PROMPT}"`);

  await chatLog(page)
    .getByText(PROMPT, { exact: false })
    .first()
    .waitFor({ state: "visible", timeout: TIMEOUT_MS.promptVisible });

  const run = await waitFor(page, {
    what: `waiting for ${N} images`,
    budgetMs: TIMEOUT_MS.deadline,
    done: (s) => s.images.length >= N || s.errored > 0,
  });
  await shot(page, "images-done");

  let s = run.snapshot;
  let tAnswer = null;
  if (s.images.length >= N) {
    pass(`${N} image(s) rendered`);
    // images are in; give the turn a shorter, separate budget to finish so a
    // hung tail still leaves us with the numbers we came for.
    const tail = await waitFor(page, {
      what: "waiting for the turn to finish",
      budgetMs: TIMEOUT_MS.tailAfterImages,
      done: (t) => !t.streaming && t.answered,
    });
    s = tail.snapshot;
    if (tail.timedOut) {
      warn(`turn did not finish within ${secs(TIMEOUT_MS.tailAfterImages)}`);
    } else {
      tAnswer = Date.now();
      pass("agent produced a final answer");
    }
  } else {
    fail(
      `only ${s.images.length} of ${N} image(s) rendered ` +
        `(${run.reason ?? "unknown"}); last state -> ${describe(s)}`,
    );
  }

  if (s.errored > 0) fail(`${s.errored} tool call(s) errored`);
  if (s.denied > 0) fail(`${s.denied} tool call(s) were denied`);
  if (run.approvals > 0) {
    warn(
      `approved ${run.approvals} gated tool call(s); the agent used bash, so ` +
        `timings include human-approval latency`,
    );
  }
  if (s.images.length > N) {
    warn(`${s.images.length} images rendered, more than the ${N} requested`);
  }
  if (s.cards.length !== N) {
    warn(`${s.cards.length} generate_image call(s) for ${N} requested images`);
  }

  if (s.images.length > 0) report(t0, s, tAnswer);
} catch (err) {
  await shot(page, "error");
  fail(`unexpected error: ${err?.stack || err}`);
} finally {
  await shot(page, "final");
  if (HEADED) {
    await new Promise((resolve) => setTimeout(resolve, TIMEOUT_MS.headedPause));
  }
  await context.close();
  await browser.close();
}

console.log("");
if (process.exitCode === 1) {
  console.error(
    `${color("red", "FAIL")} ${failures.length} problem(s) ` +
      `(see /tmp/seal-e2e-images-*.png)`,
  );
} else {
  console.log(`${color("green", "PASS")} image latency measured`);
}
