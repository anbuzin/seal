import { PlusIcon, Trash2Icon } from "lucide-react";

import type { Session } from "@/hooks/session-api";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  SidebarRail,
} from "@/components/ui/sidebar";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function groupByDate(sessions: Session[]) {
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const weekAgo = new Date(today);
  weekAgo.setDate(weekAgo.getDate() - 7);

  const groups: { label: string; items: Session[] }[] = [
    { label: "Today", items: [] },
    { label: "Yesterday", items: [] },
    { label: "Previous 7 days", items: [] },
    { label: "Older", items: [] },
  ];

  for (const s of sessions) {
    const d = new Date(s.updated_at);
    if (d.toDateString() === today.toDateString()) groups[0].items.push(s);
    else if (d.toDateString() === yesterday.toDateString())
      groups[1].items.push(s);
    else if (d >= weekAgo) groups[2].items.push(s);
    else groups[3].items.push(s);
  }

  return groups.filter((g) => g.items.length > 0);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface SessionSidebarProps {
  sessions: Session[];
  isLoading: boolean;
  currentSessionId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

export function SessionSidebar({
  sessions,
  isLoading,
  currentSessionId,
  onSelect,
  onNew,
  onDelete,
}: SessionSidebarProps) {
  const groups = groupByDate(sessions);

  return (
    <Sidebar>
      <SidebarHeader>
        <Button
          variant="ghost"
          className="w-full justify-start gap-2"
          onClick={onNew}
        >
          <PlusIcon className="size-4" />
          New chat
        </Button>
      </SidebarHeader>

      <SidebarContent>
        <ScrollArea className="flex-1">
          {isLoading ? (
            <SidebarGroup>
              <SidebarGroupContent>
                <SidebarMenu>
                  {Array.from({ length: 5 }).map((_, i) => (
                    <SidebarMenuItem key={i}>
                      <SidebarMenuSkeleton />
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          ) : sessions.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">
              No conversations yet
            </div>
          ) : (
            groups.map((group) => (
              <SidebarGroup key={group.label}>
                <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
                <SidebarGroupContent>
                  <SidebarMenu>
                    {group.items.map((session) => (
                      <SidebarMenuItem key={session.id}>
                        <SidebarMenuButton
                          isActive={session.id === currentSessionId}
                          onClick={() => onSelect(session.id)}
                          tooltip={session.title || "New conversation"}
                        >
                          <span className="truncate">
                            {session.title || "New conversation"}
                          </span>
                        </SidebarMenuButton>
                        <SidebarMenuAction
                          onClick={(e) => {
                            e.stopPropagation();
                            onDelete(session.id);
                          }}
                          showOnHover
                        >
                          <Trash2Icon className="size-4" />
                          <span className="sr-only">Delete</span>
                        </SidebarMenuAction>
                      </SidebarMenuItem>
                    ))}
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            ))
          )}
        </ScrollArea>
      </SidebarContent>

      <SidebarRail />
    </Sidebar>
  );
}
