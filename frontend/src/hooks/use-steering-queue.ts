import type { FileUIPart } from "ai";
import { useCallback, useState } from "react";
import { nanoid } from "nanoid";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SteeringItem {
  id: string;
  role: string;
  parts: Record<string, unknown>[];
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Manages a local steering queue with fire-and-forget POST to the backend.
 *
 * Items are removed when the backend emits a `data-steering-consumed`
 * DataPart via SSE (consumed from `onData` in App.tsx).
 */
export function useSteeringQueue(sessionId: string) {
  const [queue, setQueue] = useState<SteeringItem[]>([]);

  const enqueue = useCallback(
    (text: string, files?: FileUIPart[]) => {
      const id = nanoid();
      const parts: Record<string, unknown>[] = [{ type: "text", text }];
      if (files) {
        for (const f of files) {
          parts.push({
            type: "file",
            url: f.url,
            mediaType: f.mediaType,
            ...(f.filename ? { filename: f.filename } : {}),
          });
        }
      }

      const item: SteeringItem = { id, role: "user", parts };
      setQueue((prev) => [...prev, item]);

      // Fire-and-forget POST; remove on failure.
      fetch("/api/chat/steer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          messages: [{ id, role: "user", parts }],
        }),
      }).catch(() => {
        setQueue((prev) => prev.filter((i) => i.id !== id));
      });
    },
    [sessionId],
  );

  const consume = useCallback((ids: string[]) => {
    const idSet = new Set(ids);
    setQueue((prev) => prev.filter((i) => !idSet.has(i.id)));
  }, []);

  return {
    queue,
    enqueue,
    consume,
    hasItems: queue.length > 0,
  };
}
