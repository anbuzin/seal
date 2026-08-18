import { GlobeIcon } from "lucide-react"

import { Spinner } from "@/components/ui/spinner"
import { safeHttpUrl } from "@/lib/utils"
import type { WebFetchToolPart } from "@/lib/messages"

function UrlLabel({ url }: { url?: string }) {
  if (!url) return null
  const href = safeHttpUrl(url)
  if (!href) return <span className="break-all">{url}</span>
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="break-all underline underline-offset-4 hover:text-foreground"
    >
      {url}
    </a>
  )
}

export function WebFetchPart({ part }: { part: WebFetchToolPart }) {
  const url = part.input?.url

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return (
        <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
          <Spinner />
          <span>
            Fetching <UrlLabel url={url} />…
          </span>
        </div>
      )
    case "output-available":
      return (
        <>
          <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
            <GlobeIcon className="size-4" />
            <span>
              Fetched <UrlLabel url={url} />
            </span>
          </div>
          {part.output != null && (
            <pre className="max-h-64 overflow-auto rounded-lg bg-muted p-2 font-mono text-xs break-all whitespace-pre-wrap">
              {typeof part.output === "string"
                ? part.output
                : JSON.stringify(part.output, null, 2)}
            </pre>
          )}
        </>
      )
    case "output-error":
      return (
        <div className="px-1.5 text-sm text-destructive">
          Fetch failed: {part.errorText}
        </div>
      )
    default:
      return null
  }
}
