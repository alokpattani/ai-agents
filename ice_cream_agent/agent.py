from google.adk.agents import Agent
from google.adk.tools import google_search, agent_tool, FunctionTool, ToolContext
from typing import Dict

import os
from dotenv import load_dotenv

from google import genai

from google.genai.types import (
    ApiKeyConfig,
    AuthConfig,
    Content,
    GenerateContentConfig,
    GoogleMaps,
    Part,
    Tool
)

load_dotenv()

GOOGLE_GENAI_USE_VERTEXAI = os.getenv('GOOGLE_GENAI_USE_VERTEXAI')
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_CLOUD_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION') 
GOOGLE_MAPS_GROUNDING_API_KEY = os.getenv('GOOGLE_MAPS_GROUNDING_API_KEY')
ICE_CREAM_PLACES_INFO_GCS_URI = os.getenv('ICE_CREAM_PLACES_INFO_GCS_URI')

client = genai.Client(vertexai=GOOGLE_GENAI_USE_VERTEXAI, project=GOOGLE_CLOUD_PROJECT, 
    location=GOOGLE_CLOUD_LOCATION)

ice_cream_places_search_agent = Agent(
    name="ice_cream_places_search_agent",
    model="gemini-2.5-flash-preview-04-17",
    description="Agent to search for information about ice cream places using Google Search.",
    instruction="I can find information about real ice cream places using Google Search.",
    tools=[google_search]
    )

def get_my_ice_cream_places_info(prompt: str, tool_context: ToolContext) -> str:
    """Returns info about my ice cream places from CSV file containing reviews, ratings,
        and other information about the ice cream places that I've been to and documented

    Args:
        prompt (str): The prompt related to getting info about my ice cream places

    Returns:
        str: Gemini text response grounded in my ice cream places info
    """

    MY_ICE_CREAM_PLACES_MODEL_ID = "gemini-2.5-flash-preview-04-17"

    my_ice_cream_places_generation_response = client.models.generate_content(
        model=MY_ICE_CREAM_PLACES_MODEL_ID,
        contents=Content(
            role="user",
            parts=[Part.from_text(text=prompt)] + 
                [Part.from_uri(file_uri=ICE_CREAM_PLACES_INFO_GCS_URI, mime_type='text/csv')]
            ),
        config=GenerateContentConfig(
            system_instruction=(
                "You are an ice cream expert with knowledge of various ice cream places that "
                "I've been to, including my personal ratings, favorite flavors, and other notes. "
                "Please only use the provided reference material to respond about inquiries "
                "related to ice cream places. If prompts come in about ice cream places I have " 
                "not been to, respond with 'I don't have personal experience with that place.'"
                )
        )
    )

    return(my_ice_cream_places_generation_response.text)


def find_ice_cream_places(location: str, tool_context: ToolContext) -> str:
    """Returns nearby ice cream places for a specific location using Gemini 
    with grounding in Google Maps

    Args:
        location (str): The location for which to find ice cream places nearby

    Returns:
        str: Gemini text response about nearest ice cream places
    """

    google_maps_tool = Tool(
        google_maps=GoogleMaps(
            auth_config=AuthConfig(
                api_key_config=ApiKeyConfig(
                    api_key_string=GOOGLE_MAPS_GROUNDING_API_KEY
                )
            )
        )
    )

    MAPS_GROUNDING_MODEL_ID = "gemini-2.5-flash-preview-04-17"

    response = client.models.generate_content(
        model=MAPS_GROUNDING_MODEL_ID,
        contents=[
            f"Find ice cream places in or near {location}.",
            ],
        config=GenerateContentConfig(
            system_instruction=(
                "You are a helpful assistant that provides information about locations."
                "You have access to map data and can answer questions about distances, "
                "directions, and points of interest (including whether they are open or closed)."
                ),
            tools=[google_maps_tool]
        )
    )

    return response.text

ice_cream_agent = Agent(
    name="ice_cream_agent",
    model="gemini-2.5-flash-preview-04-17",
    description=(
        "Ice cream agent to find ice cream places and get information about them from Google Maps, "
        "Google Search, and my personal reviews and ratings."
    ),
    instruction=(
        "You are an ice cream agent that can find ice cream places across the world and relevant "
        "information about those places from Google Maps, Google Search, and my personal reviews "
        "and ratings. Choose the appropriate tool to respond to queries about ice cream places, "
        "trying to leverage my personal info when it's available. Feel free to use mutiple tools "
        "when responding to queries (e.g. get top ice cream places in a particular location from "
        "both Google Maps and my personal history)."
    ),
    tools=[
        agent_tool.AgentTool(agent=ice_cream_places_search_agent),
        find_ice_cream_places,
        get_my_ice_cream_places_info
        ]
    )

root_agent = ice_cream_agent
