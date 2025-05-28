from google.adk.agents import Agent
from google.adk.tools import google_search, FunctionTool, ToolContext

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
GEMINI_MODEL_ID = os.getenv('GEMINI_MODEL_ID')

client = genai.Client(vertexai=GOOGLE_GENAI_USE_VERTEXAI, project=GOOGLE_CLOUD_PROJECT, 
    location=GOOGLE_CLOUD_LOCATION)


def get_aloks_ice_cream_places_info(prompt: str, tool_context: ToolContext) -> str:
    """Returns info about Alok's ice cream places from CSV file containing favorite flavors, ratings,
        and other things about the ice cream places that Alok has been to and documented

    Args:
        prompt (str): The prompt related to getting info about Alok's ice cream places

    Returns:
        str: Gemini text response grounded in Alok's ice cream places info
    """

    aloks_ice_cream_places_generation_response = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=Content(
            role="user",
            parts=[Part.from_text(text=prompt)] + 
                [Part.from_uri(file_uri=ICE_CREAM_PLACES_INFO_GCS_URI, mime_type='text/csv')]
            ),
        config=GenerateContentConfig(
            system_instruction=(
                "You are an ice cream expert with knowledge of various ice cream places that Alok "
                "has been to, including his personal ratings, favorite flavors, and other notes. "
                "Please only use the provided reference material to respond about inquiries related "
                "to ice cream places. If prompts come in about ice cream places Alok has not been to, " 
                "respond that Alok hasn't been to that place (or any places in that area)."
                )
        )
    )

    return aloks_ice_cream_places_generation_response.text


def find_ice_cream_places(location: str, tool_context: ToolContext) -> str:
    """Returns information about ice cream places for a specific location using Gemini 
    with grounding in Google Maps

    Args:
        location (str): The location for which to find ice cream places (or information
           about them) nearby

    Returns:
        str: Gemini text response about ice cream places near the location in question
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

    response = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=[
            f"Find information about ice cream places in or near {location}.",
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
    model=GEMINI_MODEL_ID,
    description=(
        "Ice cream agent to find ice cream places and get information about them from Google Maps "
        "and Alok's personal reviews and ratings."
    ),
    instruction=(
        "You are an ice cream agent that can find ice cream places across the world and relevant "
        "information about those places from Google Maps and Alok's personal reviews and ratings. "
        "Choose the appropriate tool to respond to queries about ice cream places, trying to "
        "leverage Alok's personal info when it's available. Feel free to use both tools when "
        "responding to queries where it makes sense to have both the big-picture public perspective "
        "from Google Maps and Alok's personal curated experience (e.g. get top ice cream places in "
        "particular location from both Google Maps and Alok's personal history)."
    ),
    tools=[
        find_ice_cream_places,
        get_aloks_ice_cream_places_info
        ]
    )

root_agent = ice_cream_agent
