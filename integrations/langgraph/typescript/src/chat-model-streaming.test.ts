/**
 * Tests for the OnChatModelStream routing in handleSingleEvent.
 *
 * This suite pins the behavior to match the Python port
 * (integrations/langgraph/python/ag_ui_langgraph/agent.py, _handle_single_event):
 * mutually-exclusive routing with an early return per payload kind, plus the
 * #871 boundary block that closes an open tool call before a different one
 * starts.
 *
 * The Python port has known drops that are asserted here as the current
 * contract (a follow-up makes routing additive):
 * - text content is dropped when a tool_call_chunk is present in the same chunk
 * - a text->tool transition emits only TEXT_MESSAGE_END (the tool start is lost)
 * - trailing content on a finish_reason chunk is dropped (early return)
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { LangGraphAgent } from "./agent";
import { EventType } from "@ag-ui/client";

function createAgent() {
  const agent = new LangGraphAgent({
    graphId: "test-graph",
    deploymentUrl: "http://localhost:8000",
  });

  (agent as any).subscriber = { next: () => {} };
  (agent as any).activeRun = {
    id: "run-1",
    hasFunctionStreaming: false,
    modelMadeToolCall: false,
  };
  (agent as any).emittedToolCallStartIds = new Set<string>();
  (agent as any).messagesInProcess = {};

  const dispatched: any[] = [];
  const spy = vi.spyOn(agent as any, "dispatchEvent");
  spy.mockImplementation((event: any) => {
    dispatched.push(event);
    return true;
  });

  return { agent, dispatched };
}

type StreamChunkOpts = {
  id?: string;
  content?: any;
  toolCallChunks?: Array<{ id?: string; name?: string; args?: string }>;
  finishReason?: string | null;
  emitMessages?: boolean;
  emitToolCalls?: boolean;
  predictState?: Array<{
    tool: string;
    state_key: string;
    tool_argument: string;
  }>;
};

function streamEvent(opts: StreamChunkOpts = {}) {
  const metadata: Record<string, any> = {
    "emit-messages": opts.emitMessages ?? true,
    "emit-tool-calls": opts.emitToolCalls ?? true,
  };
  if (opts.predictState) metadata["predict_state"] = opts.predictState;

  return {
    event: "on_chat_model_stream",
    metadata,
    data: {
      chunk: {
        id: opts.id ?? "msg-1",
        content: opts.content ?? "",
        tool_call_chunks: opts.toolCallChunks,
        response_metadata: opts.finishReason
          ? { finish_reason: opts.finishReason }
          : {},
      },
    },
  };
}

const typesOf = (events: any[]) => events.map((e) => e.type);

describe("OnChatModelStream routing (Python parity)", () => {
  let agent: LangGraphAgent;
  let dispatched: any[];

  beforeEach(() => {
    ({ agent, dispatched } = createAgent());
  });

  it("emits TEXT_MESSAGE_START + CONTENT for the first text chunk", () => {
    agent.handleSingleEvent(streamEvent({ id: "msg-1", content: "Hello" }));

    expect(dispatched).toHaveLength(2);
    expect(dispatched[0]).toMatchObject({
      type: EventType.TEXT_MESSAGE_START,
      role: "assistant",
      messageId: "msg-1",
    });
    expect(dispatched[1]).toMatchObject({
      type: EventType.TEXT_MESSAGE_CONTENT,
      messageId: "msg-1",
      delta: "Hello",
    });
  });

  it("emits only CONTENT for continued text chunks", () => {
    agent.handleSingleEvent(streamEvent({ id: "msg-1", content: "Hello" }));
    agent.handleSingleEvent(streamEvent({ id: "msg-1", content: " world" }));

    expect(dispatched).toHaveLength(3);
    expect(dispatched[2]).toMatchObject({
      type: EventType.TEXT_MESSAGE_CONTENT,
      messageId: "msg-1",
      delta: " world",
    });
  });

  it("emits TOOL_CALL_ARGS after a tool call start", () => {
    agent.handleSingleEvent(
      streamEvent({
        id: "msg-1",
        toolCallChunks: [{ id: "tc-1", name: "search", args: "" }],
      }),
    );
    agent.handleSingleEvent(
      streamEvent({ id: "msg-1", toolCallChunks: [{ args: '{"query":' }] }),
    );

    expect(typesOf(dispatched)).toEqual([
      EventType.TOOL_CALL_START,
      EventType.TOOL_CALL_ARGS,
    ]);
    expect(dispatched[1]).toMatchObject({
      type: EventType.TOOL_CALL_ARGS,
      toolCallId: "tc-1",
      delta: '{"query":',
    });
  });

  it("closes the first tool call before starting a second one (#871)", () => {
    agent.handleSingleEvent(
      streamEvent({
        id: "msg-1",
        toolCallChunks: [{ id: "tc-1", name: "search", args: "" }],
      }),
    );
    // A second tool call begins while the first is still open.
    agent.handleSingleEvent(
      streamEvent({
        id: "msg-1",
        toolCallChunks: [{ id: "tc-2", name: "lookup", args: "" }],
      }),
    );

    expect(typesOf(dispatched)).toEqual([
      EventType.TOOL_CALL_START,
      EventType.TOOL_CALL_END,
      EventType.TOOL_CALL_START,
    ]);
    expect(dispatched[0]).toMatchObject({ toolCallId: "tc-1" });
    expect(dispatched[1]).toMatchObject({ toolCallId: "tc-1" });
    expect(dispatched[2]).toMatchObject({
      toolCallId: "tc-2",
      toolCallName: "lookup",
    });
  });

  // --- Known Python-port drops (a follow-up makes these additive) ---

  it("drops text content when a tool_call_chunk is present in the same chunk", () => {
    agent.handleSingleEvent(
      streamEvent({
        id: "msg-1",
        content: "Looking that up",
        toolCallChunks: [{ id: "tc-1", name: "search", args: "" }],
      }),
    );

    // Only the tool call start is emitted; the text content is lost.
    expect(typesOf(dispatched)).toEqual([EventType.TOOL_CALL_START]);
    expect(
      dispatched.some((e) => e.type === EventType.TEXT_MESSAGE_CONTENT),
    ).toBe(false);
  });

  it("drops the tool call start on a text->tool transition (emits only TEXT_MESSAGE_END)", () => {
    agent.handleSingleEvent(
      streamEvent({ id: "msg-1", content: "Let me search" }),
    );
    agent.handleSingleEvent(
      streamEvent({
        id: "msg-1",
        toolCallChunks: [{ id: "tc-1", name: "search", args: "" }],
      }),
    );

    expect(typesOf(dispatched)).toEqual([
      EventType.TEXT_MESSAGE_START,
      EventType.TEXT_MESSAGE_CONTENT,
      EventType.TEXT_MESSAGE_END,
    ]);
    expect(dispatched.some((e) => e.type === EventType.TOOL_CALL_START)).toBe(
      false,
    );
  });

  it("drops trailing content on a finish_reason chunk (early return)", () => {
    agent.handleSingleEvent(
      streamEvent({ id: "msg-1", content: "final words", finishReason: "stop" }),
    );

    expect(dispatched).toHaveLength(0);
  });

  it("emits nothing when both emit-messages and emit-tool-calls are false", () => {
    agent.handleSingleEvent(
      streamEvent({
        id: "msg-1",
        content: "ignored",
        toolCallChunks: [{ id: "tc-1", name: "search", args: "" }],
        emitMessages: false,
        emitToolCalls: false,
      }),
    );

    // The tool_call_id is still recorded for OnToolEnd dedup, but no events emit.
    expect(dispatched).toHaveLength(0);
    expect((agent as any).emittedToolCallStartIds.has("tc-1")).toBe(true);
  });
});
