import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import type { FileUIPart, ToolUIPart } from "ai";
import { Fragment, useCallback, useState } from "react";

import {
  Attachment,
  AttachmentPreview,
  AttachmentRemove,
  Attachments,
} from "@/components/ai-elements/attachments";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputActionAddAttachments,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuTrigger,
  PromptInputFooter,
  PromptInputHeader,
  PromptInputSubmit,
  PromptInputTextarea,
  usePromptInputAttachments,
} from "@/components/ai-elements/prompt-input";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import { TooltipProvider } from "@/components/ui/tooltip";

// ---------------------------------------------------------------------------
// Upload helper
// ---------------------------------------------------------------------------

async function uploadFile(file: FileUIPart): Promise<FileUIPart> {
  // Fetch the data URL content and convert to a File for upload
  const res = await fetch(file.url);
  const blob = await res.blob();
  const formData = new FormData();
  formData.append("file", blob, file.filename ?? "attachment");

  const uploadRes = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });

  if (!uploadRes.ok) {
    throw new Error(`Upload failed: ${uploadRes.statusText}`);
  }

  const { url, mediaType } = await uploadRes.json();
  return { ...file, url, mediaType };
}

// ---------------------------------------------------------------------------
// Attachment previews inside the PromptInput context
// ---------------------------------------------------------------------------

function InputAttachments() {
  const attachments = usePromptInputAttachments();

  if (attachments.files.length === 0) return null;

  return (
    <PromptInputHeader>
      <Attachments variant="inline">
        {attachments.files.map((file) => (
          <Attachment
            key={file.id}
            data={file}
            onRemove={() => attachments.remove(file.id)}
          >
            <AttachmentPreview />
            <AttachmentRemove />
          </Attachment>
        ))}
      </Attachments>
    </PromptInputHeader>
  );
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const { messages, sendMessage, status, stop } = useChat({
    transport: new DefaultChatTransport({
      api: "/api/chat",
    }),
  });

  const isLoading = status === "submitted" || status === "streaming";
  const [isUploading, setIsUploading] = useState(false);

  const handleSubmit = useCallback(
    async ({ text, files }: { text: string; files: FileUIPart[] }) => {
      if (!text.trim() && files.length === 0) return;

      // Upload files to Vercel Blob, replacing data URLs with permanent URLs
      let uploaded: FileUIPart[] = [];
      if (files.length > 0) {
        setIsUploading(true);
        try {
          uploaded = await Promise.all(files.map(uploadFile));
        } finally {
          setIsUploading(false);
        }
      }

      sendMessage({
        text,
        ...(uploaded.length > 0 ? { files: uploaded } : {}),
      });
    },
    [sendMessage],
  );

  return (
    <TooltipProvider>
      <div className="flex h-screen flex-col bg-background">
        <header className="border-b px-4 py-3">
          <div className="mx-auto w-full max-w-3xl">
            <h1 className="text-lg font-semibold">seal</h1>
          </div>
        </header>

        <Conversation className="flex-1">
          <ConversationContent>
            <div className="mx-auto w-full max-w-3xl space-y-4 px-4 py-4">
              {messages.length === 0 ? (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  <p>Send a message to start chatting</p>
                </div>
              ) : (
                messages.map((message) => (
                  <Fragment key={message.id}>
                    {message.parts.map((part, partIndex) => {
                      if (part.type.startsWith("tool-")) {
                        const toolPart = part as ToolUIPart;
                        const isComplete =
                          toolPart.state === "output-available";

                        return (
                          <Tool
                            key={`${message.id}-${partIndex}`}
                            defaultOpen={isComplete}
                          >
                            <ToolHeader
                              type={toolPart.type}
                              state={toolPart.state}
                            />
                            <ToolContent>
                              <ToolInput input={toolPart.input} />
                              <ToolOutput
                                output={toolPart.output}
                                errorText={toolPart.errorText}
                              />
                            </ToolContent>
                          </Tool>
                        );
                      }

                      if (part.type === "text") {
                        return (
                          <Message
                            key={`${message.id}-${partIndex}`}
                            from={message.role}
                          >
                            <MessageContent>
                              <MessageResponse>{part.text}</MessageResponse>
                            </MessageContent>
                          </Message>
                        );
                      }

                      if (part.type === "file") {
                        return (
                          <Message
                            key={`${message.id}-${partIndex}`}
                            from={message.role}
                          >
                            <MessageContent>
                              <Attachments variant="grid">
                                <Attachment
                                  data={{
                                    ...part,
                                    id: `${message.id}-${partIndex}`,
                                  }}
                                >
                                  <AttachmentPreview />
                                </Attachment>
                              </Attachments>
                            </MessageContent>
                          </Message>
                        );
                      }

                      return null;
                    })}
                  </Fragment>
                ))
              )}
            </div>
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        <div className="border-t p-4">
          <div className="mx-auto w-full max-w-3xl">
            <PromptInput
              accept="image/*,video/*,audio/*,application/pdf,text/*"
              multiple
              onSubmit={handleSubmit}
            >
              <InputAttachments />
              <PromptInputTextarea
                placeholder="Ask me anything..."
                disabled={isLoading || isUploading}
              />
              <PromptInputFooter>
                <PromptInputActionMenu>
                  <PromptInputActionMenuTrigger tooltip="Attach files" />
                  <PromptInputActionMenuContent>
                    <PromptInputActionAddAttachments />
                  </PromptInputActionMenuContent>
                </PromptInputActionMenu>
                <PromptInputSubmit status={status} onStop={stop} />
              </PromptInputFooter>
            </PromptInput>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
