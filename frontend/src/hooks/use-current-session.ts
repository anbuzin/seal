import { useCallback, useState } from "react";
import { nanoid } from "nanoid";

const STORAGE_KEY = "seal_session_id";

/**
 * Manages the current session ID in both React state and localStorage.
 *
 * - On mount, reads the last session ID from localStorage (if any).
 * - `select(id)` switches to an existing session.
 * - `create()` generates a new nanoid, persists it, and returns it.
 * - `clear()` resets to no active session (new-chat state).
 */
export function useCurrentSession() {
  const [sessionId, setSessionId] = useState<string | null>(() => {
    return localStorage.getItem(STORAGE_KEY);
  });

  const select = useCallback((id: string) => {
    localStorage.setItem(STORAGE_KEY, id);
    setSessionId(id);
  }, []);

  const create = useCallback((): string => {
    const id = nanoid();
    localStorage.setItem(STORAGE_KEY, id);
    setSessionId(id);
    return id;
  }, []);

  const clear = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setSessionId(null);
  }, []);

  return { sessionId, select, create, clear } as const;
}
