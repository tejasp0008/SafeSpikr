# backend/ws_test_send.py
import websocket   # websocket-client
import sys
WS_URL = "ws://127.0.0.1:8000/ws/predict"
IMG_PATH = "backend/test.jpg"  # ensure a test.jpg exists here

def on_message(ws, message):
    print("WS on_message:", message)
    ws.close()

def on_error(ws, error):
    print("WS on_error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WS closed", close_status_code, close_msg)

def on_open(ws):
    print("WS open, sending image bytes...")
    with open(IMG_PATH, "rb") as f:
        data = f.read()
        ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)

if __name__ == "__main__":
    ws = websocket.WebSocketApp(WS_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()
