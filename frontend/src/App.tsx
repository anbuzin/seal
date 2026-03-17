import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import type { FileUIPart, ToolUIPart, UIMessage } from "ai";
import { Fragment, useCallback, useEffect, useMemo, useState } from "react";

import {
  Attachment,
  AttachmentPreview,
  AttachmentRemove,
  Attachments,
} from "@/components/ai-elements/attachments";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
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
import { SessionSidebar } from "@/components/session-sidebar";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useSessionManager } from "@/hooks/use-session-manager";

// ---------------------------------------------------------------------------
// Upload helper
// ---------------------------------------------------------------------------

async function uploadFile(file: FileUIPart): Promise<FileUIPart> {
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
            className="h-14 gap-2 px-2"
            data={file}
            onRemove={() => attachments.remove(file.id)}
          >
            <AttachmentPreview className="size-10" />
            <AttachmentRemove />
          </Attachment>
        ))}
      </Attachments>
    </PromptInputHeader>
  );
}

// ---------------------------------------------------------------------------
// ChatView -- keyed by sessionId so it fully remounts on session switch
// ---------------------------------------------------------------------------

function ChatView({
  sessionId,
  initialMessages,
  onFinishReply,
}: {
  sessionId: string;
  initialMessages: UIMessage[];
  onFinishReply: () => void;
}) {
  const [isUploading, setIsUploading] = useState(false);

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
        body: { session_id: sessionId },
      }),
    [sessionId],
  );

  const { messages, sendMessage, status, stop } = useChat({
    transport,
    messages: initialMessages,
    onFinish: onFinishReply,
  });

  const isStreaming = status === "submitted" || status === "streaming";

  const handleSubmit = useCallback(
    async ({ text, files }: { text: string; files: FileUIPart[] }) => {
      if (!text.trim() && files.length === 0) return;

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
    <>
      <Conversation className="flex-1">
        <ConversationContent>
          <div className="mx-auto w-full max-w-3xl space-y-4 px-4 py-4">
            {messages.length === 0 ? (
              <ConversationEmptyState
                title="Start a conversation"
                description="Send a message to start chatting"
              />
            ) : (
              messages.map((message) => (
                <Fragment key={message.id}>
                  {message.parts.map((part, partIndex) => {
                    if (
                      typeof part.type === "string" &&
                      part.type.startsWith("tool-")
                    ) {
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

      <div className="border-t px-4 py-3">
        <div className="mx-auto w-full max-w-3xl">
          <PromptInput
            accept="image/*,video/*,audio/*,application/pdf,text/*"
            multiple
            onSubmit={handleSubmit}
          >
            <InputAttachments />
            <PromptInputTextarea
              placeholder="Ask me anything..."
              disabled={isStreaming || isUploading}
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
    </>
  );
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const mgr = useSessionManager();

  // Bootstrap on mount.
  useEffect(() => {
    mgr.bootstrap();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
          <header className="flex items-center gap-2 border-b px-4 py-3">
            <SidebarTrigger className="-ml-1" />
            <div className="mx-auto w-full max-w-3xl">
              <h1 className="text-lg font-semibold">seal</h1>
            </div>
          </header>

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
  );
}
