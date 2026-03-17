import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import type { FileUIPart, ToolUIPart, UIMessage } from "ai";
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";

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
import { SessionSidebar } from "@/components/session-sidebar";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useCurrentSession } from "@/hooks/use-current-session";
import { fetchSessionMessages } from "@/hooks/use-session-messages";
import { useSessions } from "@/hooks/use-sessions";

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
              <div className="flex h-full items-center justify-center text-muted-foreground">
                <p>Send a message to start chatting</p>
              </div>
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
  const session = useCurrentSession();
  const {
    sessions,
    isLoading: sessionsLoading,
    create,
    remove,
    generateTitle,
    invalidate,
  } = useSessions();

  const [initialMessages, setInitialMessages] = useState<UIMessage[]>([]);
  const [isReady, setIsReady] = useState(false);
  const titleTriggeredRef = useRef<string | null>(null);

  // ---- Bootstrap ---------------------------------------------------------
  // On mount: if we have a saved sessionId, load its messages.
  // If not, create a new session eagerly.

  useEffect(() => {
    (async () => {
      if (session.sessionId) {
        try {
          const msgs = await fetchSessionMessages(session.sessionId);
          setInitialMessages(msgs);
        } catch {
          // Session no longer exists; create a fresh one.
          const sid = session.create();
          await create(sid);
          setInitialMessages([]);
        }
      } else {
        const sid = session.create();
        await create(sid);
        setInitialMessages([]);
      }
      setIsReady(true);
    })();
    // Only on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- Session switching -------------------------------------------------

  const handleSelectSession = useCallback(
    async (id: string) => {
      if (id === session.sessionId) return;
      setIsReady(false);
      try {
        const msgs = await fetchSessionMessages(id);
        setInitialMessages(msgs);
        session.select(id);
      } catch {
        // ignore
      } finally {
        setIsReady(true);
      }
    },
    [session],
  );

  const handleNewSession = useCallback(async () => {
    setIsReady(false);
    const sid = session.create();
    await create(sid);
    setInitialMessages([]);
    titleTriggeredRef.current = null;
    setIsReady(true);
  }, [session, create]);

  // ---- Title generation --------------------------------------------------

  const handleFinishReply = useCallback(() => {
    const sid = session.sessionId;
    if (!sid) return;
    if (titleTriggeredRef.current === sid) return;

    const existing = sessions.find((s) => s.id === sid);
    if (existing?.title) {
      titleTriggeredRef.current = sid;
      return;
    }

    titleTriggeredRef.current = sid;
    generateTitle(sid);
    invalidate();
  }, [session.sessionId, sessions, generateTitle, invalidate]);

  // ---- Render ------------------------------------------------------------

  return (
    <TooltipProvider>
      <SidebarProvider>
        <SessionSidebar
          sessions={sessions}
          isLoading={sessionsLoading}
          currentSessionId={session.sessionId}
          onSelect={handleSelectSession}
          onNew={handleNewSession}
          onDelete={(id) => {
            remove(id);
            if (id === session.sessionId) handleNewSession();
          }}
        />

        <SidebarInset>
          <header className="flex items-center gap-2 border-b px-4 py-3">
            <SidebarTrigger className="-ml-1" />
            <div className="mx-auto w-full max-w-3xl">
              <h1 className="text-lg font-semibold">seal</h1>
            </div>
          </header>

          {!isReady || !session.sessionId ? (
            <div className="flex flex-1 items-center justify-center text-muted-foreground">
              <p>Loading...</p>
            </div>
          ) : (
            <ChatView
              key={session.sessionId}
              sessionId={session.sessionId}
              initialMessages={initialMessages}
              onFinishReply={handleFinishReply}
            />
          )}
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  );
}
