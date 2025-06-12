import os
from mistralai import Mistral
from credentials.key import key_Mistral, weather_api_key
import requests
from pydantic import BaseModel, Field
import json

api_key = key_Mistral

client = Mistral(api_key=api_key)

# https://docs.mistral.ai/capabilities/function_calling/


def get_weather(latitude, longitude):
    response = requests.get(f"https://my.meteoblue.com/packages/basic-day?apikey=fvt60WlZ3yq4aLgA&lat={latitude}&lon={longitude}&asl=42&format=json")
    return json.dumps(response.json())

def call_function(name, arguments):
    if name == "get_weather":
        args_dict = json.loads(arguments)
        return get_weather(**args_dict)
    

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather of a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "The latitude of the city"},
                    "longitude": {"type": "number", "description": "The longitude of the city"},
                },
                "required": ["latitude", "longitude"],
            }, 
        }
    }
]
#import pdb; pdb.set_trace()

messages = [{"role": "user", "content": "Quelle est la météo à Paris?"}]

    
response = client.chat.complete(
    model="mistral-large-latest",
    messages=messages,
    tools=tools,
    temperature=0,
   
)


for tool_call in response.choices[0].message.tool_calls:
    name = tool_call.function.name
    arguments = tool_call.function.arguments
    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
    result = call_function(name, arguments)
    messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})



class Reponse_meteo(BaseModel):
    temperature: float = Field(description="La température en degrés Celsius")
    response: str = Field(description="La réponse à la question")

response_2 = client.chat.complete(
    model="mistral-large-latest",
    messages=messages,
    tools=tools,
    temperature=0.3,
)

final_response = response_2.choices[0].message.content
print(final_response)




