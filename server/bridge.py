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
from livekit.agents.voice.room_io import RoomInputOptions
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
                "- `list_reminders`: show reminders. You can filter the reminders by using the following filters: today, overdue, tomorrow, no date, p1, p2, p3, p4, 7 days, next 7 days, 30 days, this week, next week, assigned to me, recurring, view all\n"
                "- `complete_reminder`: mark tasks done.\n"
                "- `start_timer`: short countdowns.\n"
                "- `search_web`: when asked about news, current events, or general web info. Be as specific as possible. Never search the web unless the user asks you to. Before searching the web, alert the user that it might take a while."
            )
        )

    # Have Ava speak first the moment the session attaches to the room.
    async def on_enter(self):
        # You can hard-code a greeting or ask the LLM to craft one.
        # Using generate_reply lets the model honor the instructions/persona.
        await self.session.generate_reply(
            instructions=(
                "Greet Davide by first name as Ava from Zurich, then briefly offer help."
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
        - content: short task text (required, non-empty)
        - due: natural language, e.g. "tomorrow 9am" or "in 2 hours"
        - project_id: optional Todoist project id
        - priority: 1..4 (4=highest)
        """
        import os, requests

        token = os.environ.get("TODOIST_TOKEN")
        assert token, "TODOIST_TOKEN not set"

        # --- sanitize inputs ---
        text = (content or "").strip()
        if not text:
            return "I need a task description (e.g., 'Call Alice')."

        prio = priority or 1
        if prio < 1:
            prio = 1
        if prio > 4:
            prio = 4

        body = {"content": text, "priority": prio}
        if due and due.strip():
            body["due_string"] = due.strip()
        if project_id and str(project_id).strip():
            body["project_id"] = str(project_id).strip()

        try:
            r = requests.post(
                "https://api.todoist.com/rest/v2/tasks",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=10,
            )
            if r.status_code >= 400:
                # surface the server's explanation to logs + return concise message
                try:
                    err = r.json()
                except Exception:
                    err = {"error": r.text}
                print(f"[todoist] create_reminder 400: payload={body} resp={err}")
                return f"Todoist rejected the request: {err.get('error') or err}"
            t = r.json()
            when = (t.get("due") or {}).get("string", "no date")
            return f"Created reminder: “{t.get('content')}” for {when} (id {t.get('id')})."
        except requests.RequestException as e:
            print(f"[todoist] create_reminder exception: {e}")
            return f"Todoist error: {e}"

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
            out.append(
                {
                    "id": t.get("id"),
                    "content": t.get("content"),
                    "due": (t.get("due") or {}).get("string"),
                    "project_id": t.get("project_id"),
                    "priority": t.get("priority"),
                    "url": t.get("url"),
                    "completed": False,  # open tasks endpoint returns only incomplete tasks
                }
            )
        return out

    @function_tool()
    async def complete_reminder(self, context: RunContext, id: str) -> str:
        """Mark a reminder (Todoist task) as done by id."""
        token = os.environ.get("TODOIST_TOKEN")
        assert token, "TODOIST_TOKEN not set"
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
            input=f"Search the web for {query}",
        )
        return resp.output_text


async def entrypoint(ctx: JobContext):
    # Connect to the LiveKit room (uses LIVEKIT_URL from env configured via the worker)
    await ctx.connect()
    print(f"[bridge] connected to room '{ctx.room.name}' as agent")

    # ---------------- Room presence gate ----------------
    # If no one is here yet, wait until someone joins before starting the agent session
    someone_present = asyncio.Event()
    if len(ctx.room.remote_participants) > 0:
        someone_present.set()

    @ctx.room.on("participant_connected")
    def _on_participant_connected(p: rtc.RemoteParticipant):
        print(f"[room] 👋 Participant joined: {p.identity}")
        if not someone_present.is_set():
            someone_present.set()

    # Wait until a participant is present
    if not someone_present.is_set():
        print("[bridge] ⏳ Waiting for a participant to join before starting session...")
        await someone_present.wait()

    # Create the agent session with OpenAI Realtime API
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            model="gpt-realtime",        # known-good realtime voice model
            # model="gpt-4o-mini-realtime",
            voice="alloy",               # documented default voice
            # voice="verse",
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
            print(f"[agent] 🔧 Tool '{call.name}' -> {out.output} (error={getattr(out, 'is_error', False)})")

    @session.on("speech_created")
    def on_speech_created(event):
        print(f"[agent] 🔊 Agent speech created (source: {event.source})")

    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        print(f"[agent] 🤖 Agent: {event.old_state} -> {event.new_state}")

    @session.on("user_state_changed")
    def on_user_state_changed(event):
        print(f"[agent] 👤 User: {event.old_state} -> {event.new_state}")

    # ---------------- Auto-shutdown when last participant leaves ----------------
    @ctx.room.on("participant_disconnected")
    async def _on_participant_disconnected(p: rtc.RemoteParticipant):
        print(f"[room] 👋 Participant left: {p.identity}")
        # If no remote participants left, close the session
        if len(ctx.room.remote_participants) == 0:
            print("[bridge] 📴 No participants left — closing agent session")
            await session.aclose()

    # ---------------- Start the session (AI will speak first via on_enter) ------
    print("[bridge] Starting voice assistant session...")
    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(close_on_disconnect=True),
    )

    print("[bridge] 🤖 Voice assistant ready!")
    print("[bridge] 💬 Try saying: 'What time is it in Zurich?' or 'Hello'")


if __name__ == "__main__":
    # Run the worker: `python bridge.py start`
    # Ensure your client subscribes and attaches the agent's remote audio track to an <audio> element.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))