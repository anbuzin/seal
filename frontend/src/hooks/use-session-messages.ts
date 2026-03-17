import type { UIMessage } from "ai";

/** Part types the UI knows how to render. */
const KNOWN_TYPES = new Set(["text", "file", "step-start", "source-url", "source-document", "reasoning"]);
function isKnownPart(p: Record<string, unknown>): boolean {
  const t = p.type as string | undefined;
  if (!t) return false;
  return KNOWN_TYPES.has(t) || t.startsWith("tool-");
}

/**
 * Fetch messages for a session and convert them to the UIMessage shape
 * that `useChat` expects as `initialMessages`.
 */
export async function fetchSessionMessages(
  sessionId: string,
): Promise<UIMessage[]> {
  const res = await fetch(`/api/sessions/${sessionId}`);
  if (!res.ok) return [];

  const data: {
    messages: { id: string; role: string; parts: Record<string, unknown>[]; createdAt: string }[];
  } = await res.json();

  return data.messages
    .map((m) => ({
      id: m.id,
      role: m.role as UIMessage["role"],
      parts: m.parts.filter(isKnownPart) as UIMessage["parts"],
      createdAt: new Date(m.createdAt),
    }))
    .filter((m) => m.parts.length > 0);
}
