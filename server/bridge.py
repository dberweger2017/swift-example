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
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from openai import OpenAI

load_dotenv()

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ROOM_NAME = os.environ.get("ROOM_NAME", "demo-room")
TODOIST_TOKEN = os.environ["TODOIST_TOKEN"]
WEB_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------- Create the Agent class with function tools ----------
class VoiceAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful voice assistant of Davide Berweger Gaillard, start the conversation by greeting the user by his first name. "
                "Your name is Ava (pronounced like in Ex Machina) and you are based in Zurich, Switzerland. "
                f"Today's date is {datetime.now(ZoneInfo('Europe/Zurich')).strftime('%H:%M:%S on %B %d, %Y')}. "
                "Always respond clearly and concisely. Talk in English unless instructed otherwise. "
                "Use the available tools instead of guessing:\n\n"
                "- `create_reminder`: add tasks or reminders.\n"
                "- `list_reminders`: show reminders.\n"
                "- `complete_reminder`: mark tasks done.\n"
                "- `start_timer`: short countdowns.\n"
                "- `search_web`: when asked about news, current events, or general web info. Be as specific as possible."
            )
        )

    @function_tool()
    async def create_reminder(
        self,
        context: RunContext,
        content: str,
        due: str | None = None,
        project_id: str | None = None,
        priority: int | None = 1,
    ) -> str:
        """
        Create a reminder (Todoist task).
        - content: short task text, e.g. "Call John".
        - due: natural language, e.g. "tomorrow 9am", "in 2 hours", or ISO date.
        - project_id: optional Todoist project id.
        - priority: 1..4 (4=highest).
        Returns a human-readable confirmation.
        """
        if not TODOIST_TOKEN:
            raise RuntimeError("TODOIST_TOKEN not configured on server")

        body = {"content": content, "priority": priority or 1}
        if due:
            body["due_string"] = due
        if project_id:
            body["project_id"] = project_id

        r = requests.post(
            "https://api.todoist.com/rest/v2/tasks",
            headers={"Authorization": f"Bearer {TODOIST_TOKEN}",
                     "Content-Type": "application/json"},
            json=body,
            timeout=10,
        )
        r.raise_for_status()
        t = r.json()
        return f"Created reminder: “{t.get('content')}” for {t.get('due', {}).get('string', 'no date')} (id {t.get('id')})."
    @function_tool()
    async def list_reminders(
        self,
        context: RunContext,
        project_id: str | None = None,
        filter: str | None = "today | overdue | tomorrow",
        limit: int | None = 10,
    ) -> list[dict]:
        """
        List upcoming reminders (Todoist tasks) and return a compact JSON list.
        - project_id: optional Todoist project id.
        - filter: Todoist query filter (e.g. "today", "overdue", "7 days").
        - limit: max tasks to return.
        Returns a list of {id, content, due, project_id, priority, url, completed}.
        """
        if not TODOIST_TOKEN:
            raise RuntimeError("TODOIST_TOKEN not configured on server")

        # Todoist supports `?project_id=` and `?filter=`
        params = {}
        if project_id:
            params["project_id"] = project_id
        if filter:
            params["filter"] = filter

        r = requests.get(
            "https://api.todoist.com/rest/v2/tasks",
            headers={"Authorization": f"Bearer {TODOIST_TOKEN}"},
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()
        out = []
        for t in items[: (limit or 10)]:
            out.append({
                "id": t.get("id"),
                "content": t.get("content"),
                "due": (t.get("due") or {}).get("string"),
                "project_id": t.get("project_id"),
                "priority": t.get("priority"),
                "url": t.get("url"),
                "completed": False,  # open tasks endpoint returns only incomplete tasks
            })
        return out

    @function_tool()
    async def complete_reminder(self, context: RunContext, id: str) -> str:
        """Mark a reminder (Todoist task) as done by id."""
        token = os.environ.get("TODOIST_TOKEN"); assert token, "TODOIST_TOKEN not set"
        r = requests.post(
            f"https://api.todoist.com/rest/v2/tasks/{id}/close",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        r.raise_for_status()
        return f"Done. Reminder {id} completed."

    @function_tool()
    async def start_timer(self, context: RunContext, seconds: int) -> str:
        """Set a short timer (<= 300s). The assistant will announce when time is up."""
        import asyncio
        if seconds < 1 or seconds > 300:
            return "Timers must be between 1 and 300 seconds."
        await context.say(f"Timer started for {seconds} seconds.")
        await asyncio.sleep(seconds)
        await context.say("Timer done.")
        return f"Timer finished after {seconds} seconds."

    @function_tool()
    async def search_web(self, context: RunContext, query: str) -> str:
        """
        Search the web for real-time information using OpenAI's web search tool.
        Returns a concise textual summary.
        """
        resp = WEB_CLIENT.responses.create(
            model="gpt-5-mini",
            tools=[{"type": "web_search"}],
            input=query,
        )
        return resp.output_text


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
