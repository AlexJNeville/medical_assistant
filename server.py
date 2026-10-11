from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ask_ai as ai 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserMessage(BaseModel):
    text: str

# --- NEW RESET ROUTE ---
@app.get("/reset")
async def reset_chat():
    try:
        # This re-initializes the session in your ask_ai script
        ai.chat = ai.client.chats.create(
            model="gemini-2.5-flash", 
            config=ai.config
        )
        print("--- Session Wiped: Ready for new user ---")
        return {"status": "success"}
    except Exception as e:
        print(f"Reset Error: {e}")
        return {"status": "error", "detail": str(e)}

# --- EXISTING CHAT ROUTE ---
@app.post("/chat")
async def chat_with_ai(message: UserMessage):
    try:
        response = ai.chat.send_message(message.text)
        final_response = ai.handle_function_calls(response)
        return {"reply": final_response.text}
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- START SERVER (MUST BE AT THE BOTTOM) ---
if __name__ == "__main__":
    import uvicorn
    # Make sure this is port 8000 to match your HTML fetch call
    uvicorn.run(app, host="127.0.0.1", port=8000)
