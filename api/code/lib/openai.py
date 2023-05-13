import openai
import os
from dotenv import load_dotenv

load_dotenv()


class OPENAI:
    def __init__(self) -> None:
        openai.api_key = os.getenv("OPENAI_APIKEY")

        systemPrompt = os.getenv("OPENAI_SYSTEM_PROMPT")

        self.messages = [
            {
                "role": "system",
                "content": systemPrompt
            }
        ]

        pass

    def ChatCompletion(self, msg) -> str:

        self.messages.append({
            "role": "user",
            "content": msg
        })

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )

        self.messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })

        return response.choices[0].message.content
