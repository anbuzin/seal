import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

// The data-* attributes are the e2e DOM contract (e2e/test.mjs): the final
// answer is detected as a trailing depth-0 assistant "message" element.
export function TextPart({
  text,
  role,
  depth,
}: {
  text: string
  role: string
  depth: number
}) {
  if (!text.trim()) return null

  return (
    <div
      data-testid="message"
      data-message-role={role}
      data-message-depth={depth}
      className="typeset typeset-docs min-w-0 px-1.5"
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  )
}
