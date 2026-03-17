/**
 * Unified session management hook.
 *
 * Combines:
 * - Current session ID tracking (localStorage + React state)
 * - Server-side session list (react-query)
 * - Session bootstrap (load messages on mount / switch)
 * - Title generation after first assistant reply
 */

import type { UIMessage } from "ai";
import { nanoid } from "nanoid";
import { useCallback, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  type Session,
  createSessionOnServer,
  deleteSessionOnServer,
  fetchSessionMessages,
  fetchSessions,
  generateSessionTitle,
} from "./session-api";

export type { Session } from "./session-api";

// ---------------------------------------------------------------------------
// localStorage persistence for current session ID
// ---------------------------------------------------------------------------

const STORAGE_KEY = "seal_session_id";

function readStoredSessionId(): string | null {
  return localStorage.getItem(STORAGE_KEY);
}

function writeStoredSessionId(id: string) {
  localStorage.setItem(STORAGE_KEY, id);
}

// ---------------------------------------------------------------------------
// Query key
// ---------------------------------------------------------------------------

const SESSIONS_KEY = ["sessions"] as const;

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSessionManager() {
  const qc = useQueryClient();

  // ---- Current session ID ------------------------------------------------
  const [sessionId, setSessionId] = useState<string | null>(readStoredSessionId);

  // ---- Messages for the active session -----------------------------------
  const [initialMessages, setInitialMessages] = useState<UIMessage[]>([]);
  const [isReady, setIsReady] = useState(false);
  const titleTriggeredRef = useRef<string | null>(null);

  // ---- Session list (server) ---------------------------------------------
  const { data: sessions = [], isLoading: sessionsLoading } = useQuery({
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

  // ---- Shared helpers ----------------------------------------------------

  /** Load messages for a session ID, updating ready/initial state. */
  const loadSession = useCallback(async (id: string) => {
    setIsReady(false);
    try {
      const msgs = await fetchSessionMessages(id);
      setInitialMessages(msgs);
      writeStoredSessionId(id);
      setSessionId(id);
    } finally {
      setIsReady(true);
    }
  }, []);

  /** Create and activate a brand-new empty session. */
  const createFreshSession = useCallback(async () => {
    const id = nanoid();
    await createMutation.mutateAsync(id);
    writeStoredSessionId(id);
    setSessionId(id);
    setInitialMessages([]);
  }, [createMutation]);

  /** Public action: create a new session (with loading state). */
  const newSession = useCallback(async () => {
    setIsReady(false);
    await createFreshSession();
    titleTriggeredRef.current = null;
    setIsReady(true);
  }, [createFreshSession]);

  // ---- Bootstrap (call once from App useEffect) --------------------------

  const bootstrap = useCallback(async () => {
    const storedId = readStoredSessionId();
    if (storedId) {
      try {
        const msgs = await fetchSessionMessages(storedId);
        setInitialMessages(msgs);
        setSessionId(storedId);
      } catch {
        await createFreshSession();
      }
    } else {
      await createFreshSession();
    }
    setIsReady(true);
  }, [createFreshSession]);

  // ---- Title generation (call from onFinish) -----------------------------

  const triggerTitle = useCallback(() => {
    if (!sessionId) return;
    if (titleTriggeredRef.current === sessionId) return;

    const existing = sessions.find((s) => s.id === sessionId);
    if (existing?.title) {
      titleTriggeredRef.current = sessionId;
      return;
    }

    titleTriggeredRef.current = sessionId;
    titleMutation.mutate(sessionId);
  }, [sessionId, sessions, titleMutation]);

  // ---- Delete ------------------------------------------------------------

  const deleteSession = useCallback(
    async (id: string) => {
      deleteMutation.mutate(id);
      if (id === sessionId) {
        await newSession();
      }
    },
    [deleteMutation, sessionId, newSession],
  );

  return {
    // State
    sessionId,
    sessions,
    sessionsLoading,
    initialMessages,
    isReady,

    // Actions
    bootstrap,
    selectSession: loadSession,
    newSession,
    deleteSession,
    triggerTitle,
  } as const;
}
