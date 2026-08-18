import { NewChatButton } from "@/components/new-chat-button"
import { SidebarTrigger } from "@/components/ui/sidebar"

export function SiteHeader({ onNewChat }: { onNewChat: () => void }) {
  return (
    <header className="flex items-center justify-between gap-2 px-6 py-3">
      <div className="flex items-center gap-2">
        <SidebarTrigger className="-ml-1" />
        <h1 className="text-sm font-medium">seal</h1>
      </div>
      <NewChatButton onNew={onNewChat} />
    </header>
  )
}
