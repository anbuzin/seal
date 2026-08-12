import { useEffect } from "react"

import { ChatView } from "@/components/chat"
import { SessionSidebar } from "@/components/session-sidebar"
import { SiteHeader } from "@/components/site-header"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import { TooltipProvider } from "@/components/ui/tooltip"
import { useSessionManager } from "@/hooks/use-session-manager"

export default function App() {
  const mgr = useSessionManager()

  // Bootstrap on mount.
  useEffect(() => {
    mgr.bootstrap()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <TooltipProvider>
      <SidebarProvider>
        <SessionSidebar
          sessions={mgr.sessions}
          isLoading={mgr.sessionsLoading}
          currentSessionId={mgr.sessionId}
          onSelect={mgr.selectSession}
          onNew={mgr.newSession}
          onDelete={mgr.deleteSession}
        />

        <SidebarInset>
          <SiteHeader onNewChat={mgr.newSession} />

          {!mgr.isReady || !mgr.sessionId ? (
            <div className="flex flex-1 items-center justify-center text-muted-foreground">
              <p>Loading...</p>
            </div>
          ) : (
            <ChatView
              key={mgr.sessionId}
              sessionId={mgr.sessionId}
              initialMessages={mgr.initialMessages}
              onFinishReply={mgr.triggerTitle}
            />
          )}
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  )
}
