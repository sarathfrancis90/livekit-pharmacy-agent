# pharmacy_agent.py

import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import elevenlabs, deepgram, openai, silero

logger = logging.getLogger("pharmacy-agent")
logger.setLevel(logging.INFO)

load_dotenv()

voices = {
    "triage": "Xb7hH8MSUJpSbSDYk0k2",
    "prescription": "nPczCjzI2devNBz1zQrb",
    "info": "XB0fDUnXU5powFXDhCwa",
}

@dataclass
class PharmacyUserData:
    customer_name: Optional[str] = None
    prescription_id: Optional[str] = None
    medicine_name: Optional[str] = None
    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        return yaml.dump({
            "customer_name": self.customer_name or "unknown",
            "prescription_id": self.prescription_id or "unknown",
            "medicine_name": self.medicine_name or "unknown",
        })

RunContext_T = RunContext[PharmacyUserData]

@function_tool()
async def update_name(name: Annotated[str, Field(description="The customer's name")], context: RunContext_T) -> str:
    context.userdata.customer_name = name
    return f"Updated your name to {name}."

@function_tool()
async def check_prescription_status(prescription_id: Annotated[str, Field(...)], context: RunContext_T) -> str:
    context.userdata.prescription_id = prescription_id
    return f"Prescription {prescription_id} is ready for pickup. (mocked)"

@function_tool()
async def check_medicine_availability(medicine: Annotated[str, Field(...)], context: RunContext_T) -> str:
    context.userdata.medicine_name = medicine
    return f"{medicine} is currently in stock. (mocked)"

@function_tool()
async def get_pharmacy_info(context: RunContext_T) -> str:
    return (
        "Our pharmacy is located at 123 Health Ave, Wellness City.\n"
        "We are open Monday to Friday from 9 AM to 7 PM, and Saturday from 10 AM to 4 PM."
    )

@function_tool()
async def to_triage(context: RunContext_T) -> Agent:
    return await context.session.current_agent._transfer_to_agent("triage", context)

class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name}")
        userdata = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        if isinstance(userdata.prev_agent, Agent):
            previous = userdata.prev_agent.chat_ctx.copy(exclude_instructions=True).truncate(6)
            chat_ctx.items.extend(i for i in previous.items if i.id not in {x.id for x in chat_ctx.items})

        chat_ctx.add_message("system", f"You are {agent_name}. Current user data:\n{userdata.summarize()}")
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        next_agent = userdata.agents[name]
        userdata.prev_agent = context.session.current_agent
        return next_agent, f"Transferring to {name}."

class TriageAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            instructions="You are the triage agent for a pharmacy. Greet the caller and find out if they need prescription status, medicine availability, or general info. Use tools to route or answer directly.",
            tools=[check_prescription_status, check_medicine_availability, get_pharmacy_info],
            tts=elevenlabs.TTS(voice_id=voices["triage"], model="eleven_multilingual_v2"),
        )

class PrescriptionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            instructions="You handle prescription-related questions like status, refills, and pickup details. Mock responses.",
            tools=[update_name, check_prescription_status, to_triage],
            tts=elevenlabs.TTS(voice_id=voices["prescription"], model="eleven_multilingual_v2"),
        )

class InfoAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            instructions="You provide general pharmacy information like hours, address, or services.",
            tools=[get_pharmacy_info, to_triage],
            tts=elevenlabs.TTS(voice_id=voices["info"], model="eleven_multilingual_v2"),
        )

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    userdata = PharmacyUserData()
    userdata.agents.update({
        "triage": TriageAgent(),
        "prescription": PrescriptionAgent(),
        "info": InfoAgent(),
    })

    session = AgentSession[PharmacyUserData](
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM.with_cerebras(
        model="llama3.1-8b",
        temperature=0.7
         ),
        tts=elevenlabs.TTS(voice_id=voices["info"], model="eleven_multilingual_v2"),
        vad=silero.VAD.load(),
        max_tool_steps=5,
    )

    await session.start(
        agent=userdata.agents["triage"],
        room=ctx.room,
        room_input_options=RoomInputOptions()
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
