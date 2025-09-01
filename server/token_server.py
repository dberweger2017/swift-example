# server/token_server.py
import os
from datetime import timedelta
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

load_dotenv()

API_KEY = os.environ["LIVEKIT_API_KEY"]
API_SECRET = os.environ["LIVEKIT_API_SECRET"]

app = Flask(__name__)

@app.post("/token")
def token():
    body = request.get_json(force=True) or {}
    identity = body.get("identity", "ios-user")
    room = body.get("room", "demo-room")

    grants = VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )

    token = (
        AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_ttl(timedelta(hours=1))
        .with_grants(grants)
        .to_jwt()
    )

    return jsonify({"token": token, "room": room})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8787)
