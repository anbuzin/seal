import { PlusIcon } from "lucide-react"

import { Button } from "@/components/ui/button"

// The template's version reloads the page via an anchor; ours starts a new
// session through the session manager instead.
export function NewChatButton({ onNew }: { onNew: () => void }) {
  return (
    <Button variant="secondary" onClick={onNew}>
      <PlusIcon data-icon="inline-start" />
      New Chat
    </Button>
  )
}
