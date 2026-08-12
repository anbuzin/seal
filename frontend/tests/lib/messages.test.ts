import { describe, expect, it } from "vitest"

import { getFreshParts } from "../../src/lib/messages"

const text = (i: number) => ({ type: "text", i })
const tool = (toolCallId: string, i = 0) => ({
  type: "tool-bash",
  toolCallId,
  i,
})

describe("getFreshParts", () => {
  it("drops the previous step on data-reload", () => {
    expect(
      getFreshParts([
        { type: "step-start" },
        text(1),
        { type: "data-reload" },
        text(2),
      ])
    ).toEqual([{ type: "step-start" }, text(2)])
  })

  it("drops everything on data-reload when there is no step boundary", () => {
    // seeded reload history carries no step-start parts; the replayed turn
    // replaces it wholesale.
    expect(getFreshParts([text(1), { type: "data-reload" }, text(2)])).toEqual([
      text(2),
    ])
  })

  it("dedupes replayed tool parts by toolCallId, keeping the last", () => {
    // AI SDK v7 scopes tool reconciliation to the current step, so a replayed
    // turn appends a second copy of each seeded tool part (pinned in
    // tests/contract.test.ts). Render must keep only the replayed copy.
    expect(
      getFreshParts([
        text(1),
        tool("tc-a", 1),
        tool("tc-b", 1),
        { type: "step-start" },
        tool("tc-a", 2),
        tool("tc-b", 2),
      ])
    ).toEqual([
      text(1),
      { type: "step-start" },
      tool("tc-a", 2),
      tool("tc-b", 2),
    ])
  })

  it("keeps distinct tool calls and dedupes dynamic tools too", () => {
    expect(
      getFreshParts([
        { type: "dynamic-tool", toolCallId: "tc-a", i: 1 },
        tool("tc-b"),
        { type: "dynamic-tool", toolCallId: "tc-a", i: 2 },
      ])
    ).toEqual([
      tool("tc-b"),
      { type: "dynamic-tool", toolCallId: "tc-a", i: 2 },
    ])
  })
})
