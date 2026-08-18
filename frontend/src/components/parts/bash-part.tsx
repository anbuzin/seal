import type { ChatAddToolApproveResponseFunction } from "ai"
import { BanIcon, ShieldAlertIcon, TerminalIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import type { BashToolPart } from "@/lib/messages"

function Command({ command }: { command?: string }) {
  if (!command) return null
  return (
    <pre className="max-h-24 overflow-y-auto px-1.5 font-mono text-xs break-all whitespace-pre-wrap text-muted-foreground">
      {command}
    </pre>
  )
}

export function BashPart({
  part,
  onApprovalResponse,
}: {
  part: BashToolPart
  onApprovalResponse: ChatAddToolApproveResponseFunction
}) {
  const command = part.input?.command

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return (
        <>
          <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
            <Spinner />
            Running command…
          </div>
          <Command command={command} />
        </>
      )
    case "approval-requested":
      return (
        <>
          <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
            <ShieldAlertIcon className="size-4" />
            Approve running this command?
          </div>
          <Command command={command} />
          {part.approval && (
            <div className="flex items-center gap-2 px-1.5 py-1">
              <Button
                size="sm"
                variant="outline"
                onClick={() =>
                  onApprovalResponse({ id: part.approval!.id, approved: false })
                }
              >
                Reject
              </Button>
              <Button
                size="sm"
                onClick={() =>
                  onApprovalResponse({ id: part.approval!.id, approved: true })
                }
              >
                Approve
              </Button>
            </div>
          )}
        </>
      )
    case "approval-responded":
      return (
        <>
          <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
            {part.approval?.approved ? (
              <>
                <Spinner />
                Approved, running…
              </>
            ) : (
              <>
                <BanIcon className="size-4" />
                Rejected
              </>
            )}
          </div>
          <Command command={command} />
        </>
      )
    case "output-available":
      return (
        <>
          <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
            <TerminalIcon className="size-4" />
            Ran command
          </div>
          <Command command={command} />
          {part.output != null && (
            <pre className="max-h-64 overflow-auto rounded-lg bg-muted p-2 font-mono text-xs break-all whitespace-pre-wrap">
              {typeof part.output === "string"
                ? part.output
                : JSON.stringify(part.output, null, 2)}
            </pre>
          )}
        </>
      )
    case "output-denied":
      return (
        <>
          <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
            <BanIcon className="size-4" />
            Command denied
          </div>
          <Command command={command} />
        </>
      )
    case "output-error":
      return (
        <>
          <Command command={command} />
          <div className="px-1.5 text-sm text-destructive">
            Command failed: {part.errorText}
          </div>
        </>
      )
    default:
      return null
  }
}
