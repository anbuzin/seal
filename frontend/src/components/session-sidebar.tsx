import { cn } from "@/lib/utils";
import { MessageSquarePlus, Trash2, PanelLeftClose, PanelLeft } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

export interface Session {
  id: string;
  user_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

interface SessionSidebarProps {
  currentSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
  className?: string;
}

export function SessionSidebar({
  currentSessionId,
  onSelectSession,
  onNewSession,
  className,
}: SessionSidebarProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCollapsed, setIsCollapsed] = useState(false);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch("/api/sessions", { credentials: "include" });
      if (res.ok) {
        const data = await res.json();
        setSessions(data.sessions);
      }
    } catch (error) {
      console.error("Failed to fetch sessions:", error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // Refresh sessions when currentSessionId changes (new session created)
  useEffect(() => {
    if (currentSessionId) {
      fetchSessions();
    }
  }, [currentSessionId, fetchSessions]);

  const handleDeleteSession = useCallback(
    async (e: React.MouseEvent, sessionId: string) => {
      e.stopPropagation();
      try {
        const res = await fetch(`/api/sessions/${sessionId}`, {
          method: "DELETE",
          credentials: "include",
        });
        if (res.ok) {
          setSessions((prev) => prev.filter((s) => s.id !== sessionId));
          if (currentSessionId === sessionId) {
            onNewSession();
          }
        }
      } catch (error) {
        console.error("Failed to delete session:", error);
      }
    },
    [currentSessionId, onNewSession]
  );

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  if (isCollapsed) {
    return (
      <div className={cn("flex flex-col border-r bg-card w-12", className)}>
        <div className="p-2 border-b">
          <button
            onClick={() => setIsCollapsed(false)}
            className="p-2 hover:bg-accent transition-colors w-full flex justify-center"
            title="Expand sidebar"
          >
            <PanelLeft className="size-4" />
          </button>
        </div>
        <div className="p-2">
          <button
            onClick={onNewSession}
            className="p-2 hover:bg-accent transition-colors w-full flex justify-center"
            title="New chat"
          >
            <MessageSquarePlus className="size-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col border-r bg-card w-64", className)}>
      <div className="flex items-center justify-between p-3 border-b">
        <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          Sessions
        </span>
        <button
          onClick={() => setIsCollapsed(true)}
          className="p-1 hover:bg-accent transition-colors"
          title="Collapse sidebar"
        >
          <PanelLeftClose className="size-4" />
        </button>
      </div>

      <div className="p-2">
        <button
          onClick={onNewSession}
          className="flex items-center gap-2 w-full p-2 text-sm hover:bg-accent transition-colors border border-dashed border-border"
        >
          <MessageSquarePlus className="size-4" />
          <span>New chat</span>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-3 text-sm text-muted-foreground">Loading...</div>
        ) : sessions.length === 0 ? (
          <div className="p-3 text-sm text-muted-foreground">
            No sessions yet
          </div>
        ) : (
          <div className="space-y-1 p-2">
            {sessions.map((session) => (
              <div
                key={session.id}
                onClick={() => onSelectSession(session.id)}
                className={cn(
                  "group flex items-center justify-between p-2 text-sm cursor-pointer transition-colors",
                  currentSessionId === session.id
                    ? "bg-accent text-accent-foreground"
                    : "hover:bg-accent/50"
                )}
              >
                <div className="flex-1 min-w-0">
                  <div className="truncate">
                    {session.title || "New conversation"}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {formatDate(session.updated_at)}
                  </div>
                </div>
                <button
                  onClick={(e) => handleDeleteSession(e, session.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/20 hover:text-destructive transition-all"
                  title="Delete session"
                >
                  <Trash2 className="size-3" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
