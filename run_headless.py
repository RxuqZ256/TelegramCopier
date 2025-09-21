import os, asyncio
from telethon import TelegramClient, events

api_id  = int(os.getenv("TG_API_ID","0"))
api_hash= os.getenv("TG_API_HASH","")
phone   = os.getenv("TG_PHONE","")

client = TelegramClient("trading_session", api_id, api_hash)

async def main():
    await client.connect()
    if not await client.is_user_authorized():
        await client.send_code_request(phone)
        code = input("Telegram-Code eingeben: ")
        await client.sign_in(phone=phone, code=code)

    print("✅ Verbunden. Warte auf Nachrichten…")
    @client.on(events.NewMessage)
    async def handler(e):
        msg = (e.message.message or "").replace("\n"," ")[:120]
        print(f"[{e.chat_id}] {msg}")

    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())
