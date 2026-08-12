import type { FileUIPart } from "ai"
import {
  ArrowUpIcon,
  FileIcon,
  PaperclipIcon,
  SquareIcon,
  XIcon,
} from "lucide-react"
import * as React from "react"

import {
  Attachment,
  AttachmentAction,
  AttachmentActions,
  AttachmentContent,
  AttachmentGroup,
  AttachmentMedia,
  AttachmentTitle,
} from "@/components/ui/attachment"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupTextarea,
} from "@/components/ui/input-group"
import { ModelSelect } from "@/components/model-select"
import type { GatewayModel } from "@/lib/models"

const ACCEPT = "image/*,video/*,audio/*,application/pdf,text/*"

function readAsFilePart(file: File): Promise<FileUIPart> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () =>
      resolve({
        type: "file",
        mediaType: file.type || "application/octet-stream",
        filename: file.name,
        url: reader.result as string,
      })
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

export function PromptForm({
  models,
  model,
  onModelChange,
  isBusy,
  onSubmit,
  onStop,
}: {
  models: GatewayModel[]
  model: string
  onModelChange: (model: string) => void
  isBusy: boolean
  onSubmit: (message: { text: string; files: FileUIPart[] }) => void
  onStop: () => void
}) {
  const [input, setInput] = React.useState("")
  const [files, setFiles] = React.useState<FileUIPart[]>([])
  const fileInputRef = React.useRef<HTMLInputElement>(null)

  function handleSubmit(event?: React.FormEvent) {
    event?.preventDefault()
    const text = input.trim()
    if ((!text && files.length === 0) || isBusy) return
    onSubmit({ text, files })
    setInput("")
    setFiles([])
  }

  async function addFiles(list: FileList | null) {
    if (!list || list.length === 0) return
    const added = await Promise.all(Array.from(list).map(readAsFilePart))
    setFiles((prev) => [...prev, ...added])
  }

  return (
    <form onSubmit={handleSubmit}>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={ACCEPT}
        className="hidden"
        onChange={(event) => {
          void addFiles(event.target.files)
          event.target.value = ""
        }}
      />
      <InputGroup>
        {files.length > 0 && (
          <InputGroupAddon align="block-start">
            <AttachmentGroup>
              {files.map((file, index) => (
                <Attachment key={index} size="xs">
                  {file.mediaType.startsWith("image/") ? (
                    <AttachmentMedia variant="image">
                      <img src={file.url} alt={file.filename} />
                    </AttachmentMedia>
                  ) : (
                    <AttachmentMedia>
                      <FileIcon />
                    </AttachmentMedia>
                  )}
                  <AttachmentContent>
                    <AttachmentTitle>{file.filename}</AttachmentTitle>
                  </AttachmentContent>
                  <AttachmentActions>
                    <AttachmentAction
                      aria-label={`Remove ${file.filename}`}
                      onClick={() =>
                        setFiles((prev) => prev.filter((_, i) => i !== index))
                      }
                    >
                      <XIcon />
                    </AttachmentAction>
                  </AttachmentActions>
                </Attachment>
              ))}
            </AttachmentGroup>
          </InputGroupAddon>
        )}
        <InputGroupTextarea
          placeholder="Ask me anything..."
          className="p-3.5"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault()
              handleSubmit()
            }
          }}
        />
        <InputGroupAddon align="block-end">
          <InputGroupButton
            type="button"
            size="icon-sm"
            variant="ghost"
            aria-label="Attach files"
            onClick={() => fileInputRef.current?.click()}
          >
            <PaperclipIcon />
          </InputGroupButton>
          <ModelSelect
            models={models}
            value={model}
            onValueChange={onModelChange}
          />
          {isBusy ? (
            <InputGroupButton
              type="button"
              size="icon-sm"
              variant="outline"
              aria-label="Stop"
              className="ml-auto"
              onClick={onStop}
            >
              <SquareIcon />
            </InputGroupButton>
          ) : (
            <InputGroupButton
              type="submit"
              size="icon-sm"
              variant="default"
              aria-label="Submit"
              className="ml-auto"
              disabled={!input.trim() && files.length === 0}
            >
              <ArrowUpIcon />
            </InputGroupButton>
          )}
        </InputGroupAddon>
      </InputGroup>
    </form>
  )
}
