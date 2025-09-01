import os
import asyncio
from typing import Any, Dict

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import openai

load_dotenv()

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ROOM_NAME = os.environ.get("ROOM_NAME", "demo-room")

# ---------- Create the Agent class with function tools ----------
class VoiceAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. "
                "When asked about time in Zurich, use the get_time_in_zurich function. "
                "Always respond clearly and concisely. Talk in english unless instructed otherwise."
            )
        )

    @function_tool()
    async def get_time_in_zurich(self, context: RunContext) -> str:
        """Get the current time in Zurich, Switzerland"""
        from datetime import datetime
        from zoneinfo import ZoneInfo  # stdlib (no external deps)

        now = datetime.now(ZoneInfo("Europe/Zurich"))
        result = f"The current time in Zurich is {now.strftime('%H:%M:%S on %B %d, %Y')}"
        print(f"[tool] 🕐 get_time_in_zurich called, returning: {result}")
        return result


async def entrypoint(ctx: JobContext):
    # Connect to the LiveKit room (uses LIVEKIT_URL from env configured via the worker)
    await ctx.connect()
    print(f"[bridge] connected to room '{ctx.room.name}' as agent")

    # Create the agent session with OpenAI Realtime API
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            model="gpt-realtime",        # known-good realtime voice model
            #model="gpt-4o-mini-realtime",
            voice="alloy",               # documented default voice
            #voice="verse",
            api_key=OPENAI_API_KEY,
            temperature=0.7,
        ),
    )

    # ---------------- AgentSession events (server ↔ LLM) ----------------
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event):
        print(f"[agent] 👤 User said: '{event.transcript}'")

    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        # Logs both user and assistant textual items as they appear
        if getattr(event.item, "text_content", None):
            print(f"[agent] 💬 {event.item.role}: {event.item.text_content}")

    @session.on("function_tools_executed")
    def on_function_tools_executed(event):
        print("[agent] 🔧 Function tools executed!")
        for call, out in event.zipped():
            # FunctionCallOutput fields: name, output, is_error, etc.
            print(f"[agent] 🔧 Tool '{call.name}' -> {out.output} (error={getattr(out, 'is_error', False)})")

    @session.on("speech_created")
    def on_speech_created(event):
        # This indicates the model generated audio for playback to the room
        print(f"[agent] 🔊 Agent speech created (source: {event.source})")

    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        print(f"[agent] 🤖 Agent: {event.old_state} -> {event.new_state}")

    @session.on("user_state_changed")
    def on_user_state_changed(event):
        print(f"[agent] 👤 User: {event.old_state} -> {event.new_state}")

    # ---------------- Room events (server ↔ participants) ----------------
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        print(f"[room] 👋 Participant joined: {participant.identity}")

    @ctx.room.on("track_published")
    def on_track_published(publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        print(f"[room] 📢 {participant.identity} published {publication.kind} track")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        print(f"[room] 🎵 Subscribed to {publication.kind} from {participant.identity}")

    # Start the session — this wires RoomIO so the agent will publish audio to the room
    print("[bridge] Starting voice assistant session...")
    await session.start(agent=VoiceAssistant(), room=ctx.room)

    print("[bridge] 🤖 Voice assistant ready!")
    print("[bridge] 💬 Try saying: 'What time is it in Zurich?' or 'Hello'")


if __name__ == "__main__":
    # Run the worker: `python bridge.py start`
    # Ensure your client subscribes and attaches the agent's remote audio track to an <audio> element.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
