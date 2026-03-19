import type { FileUIPart } from "ai";
import { useCallback } from "react";
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { nanoid } from "nanoid";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SteeringItem {
  id: string;
  session_id: string;
  role: string;
  parts: Record<string, unknown>[];
  created_at: string;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function fetchSteeringQueue(
  sessionId: string,
): Promise<SteeringItem[]> {
  const res = await fetch(`/api/sessions/${sessionId}/steering`);
  if (!res.ok) return [];
  return res.json();
}

async function postSteer(
  sessionId: string,
  id: string,
  parts: Record<string, unknown>[],
): Promise<void> {
  const res = await fetch("/api/chat/steer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      messages: [{ id, role: "user", parts }],
    }),
  });
  if (!res.ok) throw new Error("Steer request failed");
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

const STEERING_KEY = (id: string) => ["steering", id] as const;

/**
 * Manages a DB-backed steering queue with optimistic updates.
 *
 * Polls the backend while `shouldSteer` is true so the UI stays
 * in sync as the agent consumes items.
 */
export function useSteeringQueue(
  sessionId: string,
  shouldSteer: boolean,
) {
  const qc = useQueryClient();
  const key = STEERING_KEY(sessionId);

  // Poll the backend queue while we're in steering mode.
  const { data: queue = [] } = useQuery({
    queryKey: key,
    queryFn: () => fetchSteeringQueue(sessionId),
    refetchInterval: shouldSteer ? 1000 : false,
    enabled: shouldSteer,
  });

  const mutation = useMutation({
    mutationFn: ({
      id,
      parts,
    }: {
      id: string;
      parts: Record<string, unknown>[];
    }) => postSteer(sessionId, id, parts),

    // Optimistic update: append immediately.
    onMutate: async ({ id, parts }) => {
      await qc.cancelQueries({ queryKey: key });
      const previous = qc.getQueryData<SteeringItem[]>(key) ?? [];
      const optimistic: SteeringItem = {
        id,
        session_id: sessionId,
        role: "user",
        parts,
        created_at: new Date().toISOString(),
      };
      qc.setQueryData<SteeringItem[]>(key, [...previous, optimistic]);
      return { previous };
    },
    onError: (_err, _vars, context) => {
      // Rollback on failure.
      if (context?.previous) {
        qc.setQueryData<SteeringItem[]>(key, context.previous);
      }
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: key });
    },
  });

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
      mutation.mutate({ id, parts });
    },
    [mutation],
  );

  return {
    queue,
    enqueue,
    hasItems: queue.length > 0,
  };
}
