import { ImageIcon } from "lucide-react"

import { Spinner } from "@/components/ui/spinner"
import type { GenerateImageToolPart } from "@/lib/messages"

// generate_image returns multipart output: a JSON array of part dumps like
// [{kind: "text", text}, {kind: "file", data: <base64>, media_type: "image/…"}].
type OutputPart = {
  kind: string
  text?: string
  data?: string
  media_type?: string
}

export function GenerateImagePart({ part }: { part: GenerateImageToolPart }) {
  const prompt = part.input?.prompt ? ` “${part.input.prompt}”` : ""

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return (
        <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
          <Spinner />
          Generating an image{prompt}…
        </div>
      )
    case "output-available": {
      const outputParts = Array.isArray(part.output)
        ? (part.output as OutputPart[])
        : []
      const images = outputParts.filter(
        (p) => p.kind === "file" && p.media_type?.startsWith("image/")
      )
      const texts = outputParts.filter((p) => p.kind === "text" && p.text)
      return (
        <>
          <div className="flex items-center gap-2 px-1.5 text-sm text-muted-foreground">
            <ImageIcon className="size-4" />
            Generated an image{prompt}
          </div>
          {images.length > 0 && (
            <div className="flex min-w-0 flex-col gap-2">
              {images.map((img, index) => (
                <img
                  key={index}
                  src={`data:${img.media_type};base64,${img.data}`}
                  alt="Generated image"
                  className="max-w-full rounded-lg"
                />
              ))}
              {texts.map((t, index) => (
                <p key={index} className="text-xs text-muted-foreground">
                  {t.text}
                </p>
              ))}
            </div>
          )}
        </>
      )
    }
    case "output-error":
      return (
        <div className="px-1.5 text-sm text-destructive">
          Image generation failed: {part.errorText}
        </div>
      )
    default:
      return null
  }
}
