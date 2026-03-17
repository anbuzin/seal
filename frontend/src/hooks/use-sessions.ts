import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export interface Session {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

const SESSIONS_KEY = ["sessions"] as const;

async function fetchSessions(): Promise<Session[]> {
  const res = await fetch("/api/sessions");
  if (!res.ok) throw new Error("Failed to fetch sessions");
  return res.json();
}

async function createSessionOnServer(id: string): Promise<Session> {
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) throw new Error("Failed to create session");
  return res.json();
}

async function deleteSessionOnServer(id: string): Promise<void> {
  const res = await fetch(`/api/sessions/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete session");
}

async function generateSessionTitle(
  id: string,
): Promise<Session> {
  const res = await fetch(`/api/sessions/${id}/title`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to generate title");
  return res.json();
}

/**
 * React-query based session management.
 *
 * Provides `sessions` list, plus `create`, `remove`, and
 * `generateTitle` mutations that optimistically update the cache.
 */
export function useSessions() {
  const qc = useQueryClient();

  const { data: sessions = [], isLoading } = useQuery({
    queryKey: SESSIONS_KEY,
    queryFn: fetchSessions,
  });

  const createMutation = useMutation({
    mutationFn: createSessionOnServer,
    onSuccess: (session) => {
      qc.setQueryData<Session[]>(SESSIONS_KEY, (old = []) => [session, ...old]);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteSessionOnServer,
    onMutate: async (id) => {
      await qc.cancelQueries({ queryKey: SESSIONS_KEY });
      qc.setQueryData<Session[]>(SESSIONS_KEY, (old = []) =>
        old.filter((s) => s.id !== id),
      );
    },
    onSettled: () => qc.invalidateQueries({ queryKey: SESSIONS_KEY }),
  });

  const titleMutation = useMutation({
    mutationFn: generateSessionTitle,
    onSuccess: (updated) => {
      qc.setQueryData<Session[]>(SESSIONS_KEY, (old = []) =>
        old.map((s) => (s.id === updated.id ? updated : s)),
      );
    },
  });

  return {
    sessions,
    isLoading,
    create: createMutation.mutateAsync,
    remove: deleteMutation.mutate,
    generateTitle: titleMutation.mutate,
    invalidate: () => qc.invalidateQueries({ queryKey: SESSIONS_KEY }),
  } as const;
}
