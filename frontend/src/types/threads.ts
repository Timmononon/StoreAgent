import type { RagTrace } from './chat';

export type ThreadMessageRole = 'user' | 'assistant' | 'system';

export interface ThreadSummary {
  thread_id: string;
  title: string;
  message_count: number;
  updated_at: string;
  // version 保护消息追加顺序，已有 assistant Message 的正文更新不推进它。
  version: number;
  thread_status: string;
  active_run_id: string | null;
  active_run_status: string | null;
}

export interface ThreadDetail extends ThreadSummary {
  created_at: string;
}

export interface ThreadListItem extends ThreadSummary {
  activeRunId: string | null;
  activeRunStatus: string | null;
  isStreaming: boolean;
}

export interface ThreadMessage {
  id: number;
  run_id: string | null;
  sequence: number;
  status: string;
  role: ThreadMessageRole;
  content: string;
  timestamp: string;
  rag_trace: RagTrace | null;
  skill_name?: string | null;
}

export interface ThreadMessagePage {
  messages: ThreadMessage[];
  // 使用 Message sequence 获取更早消息；null 表示没有上一页。
  previous_cursor: number | null;
}

export interface ThreadCreateRequest {
  title?: string | null;
}

export interface ThreadDeleteResult {
  thread_id: string;
  message: string;
}
