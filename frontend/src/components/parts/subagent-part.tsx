import { BotIcon } from "lucide-react"
import type { ReactNode } from "react"

import { Spinner } from "@/components/ui/spinner"
import type { SubagentToolPart } from "@/lib/messages"

// The nested conversation is rendered by chat-message.tsx (which owns the
// part dispatcher) and passed in as children; when the child agent's final
// output is just a text summary, fall back to rendering that.
export function SubagentPart({
  part,
  children,
}: {
  part: SubagentToolPart
  children?: ReactNode
}) {
  const name = part.input?.name || "subagent"

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return (
        <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
          <Spinner />
          Delegating to {name}…
        </div>
      )
    case "output-available": {
      const running = part.preliminary
      return (
        <>
          <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
            {running ? <Spinner /> : <BotIcon className="size-4" />}
            {running ? <>Delegating to {name}…</> : <>Delegated to {name}</>}
          </div>
          {children ? (
            <div className="flex min-w-0 flex-col gap-2 border-l pl-3">
              {children}
            </div>
          ) : typeof part.output === "string" ? (
            <pre className="max-h-64 overflow-auto rounded-lg bg-muted p-2 font-mono text-xs break-all whitespace-pre-wrap">
              {part.output}
            </pre>
          ) : null}
        </>
      )
    }
    case "output-error":
      return (
        <div className="px-1.5 text-sm text-destructive">
          Subagent failed: {part.errorText}
        </div>
      )
    default:
      return null
  }
}
