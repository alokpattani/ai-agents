from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext

import os
from dotenv import load_dotenv

from google import genai
from google.genai.types import (
    ApiKeyConfig,
    AuthConfig,
    Content,
    GenerateContentConfig,
    GenerateImagesConfig,
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
IMAGEN_MODEL_ID = os.getenv('IMAGEN_MODEL_ID')

client = genai.Client(vertexai=GOOGLE_GENAI_USE_VERTEXAI,
    project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)


async def get_aloks_ice_cream_places_info(prompt: str, tool_context: 
    ToolContext) -> str:
    """Returns info about Alok's ice cream places from file containing
       favorite flavors, ratings, and other things about the ice cream places 
       that Alok has been to in the US

    Args:
        prompt (str): Prompt to get info about Alok's ice cream places

    Returns:
        str: Gemini response grounded in Alok's ice cream places info
    """

    aloks_ice_cream_places_generation_response = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=Content(
            role="user",
            parts=[Part.from_text(text=prompt)] + 
                [Part.from_uri(file_uri=ICE_CREAM_PLACES_INFO_GCS_URI, 
                mime_type='text/csv')]
            ),
        config=GenerateContentConfig(
            system_instruction=("""
                You are an ice cream expert with knowledge of various ice cream 
                places that Alok has been to, including his personal ratings,
                favorite flavors, and other notes. Please only use the provided
                reference material to respond about inquiries related to ice 
                cream places.

                If a place is marked as TRUE in the locationKnownClosed field in
                Alok's list, do not use it in a response about current ice cream
                places, and note that it's closed if you do use it in a
                particular response.
                
                If prompts come in about ice cream places Alok has not been to,
                respond that Alok hasn't been to that place (or any places in
                that area).
                """)
        )
    )

    return aloks_ice_cream_places_generation_response.text


async def get_google_maps_ice_cream_places_info(prompt: str,
    tool_context: ToolContext) -> str:
    """Returns info about ice cream places using Gemini grounded in Google Maps

    Args:
        prompt (str): Gemini prompt to get ice cream places info using Maps
          
    Returns:
        str: Gemini response about ice cream places using grounding in Maps
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
        contents=[prompt],
        config=GenerateContentConfig(
            system_instruction=("""
                You are a helpful assistant that provides information about ice
                cream places. You have access to map data and can answer
                questions about distances, directions, and points of interest 
                (including whether they are currently open or closed).
                """
                ),
            tools=[google_maps_tool]
        )
    )

    return response.text


async def generate_ice_cream_image(prompt: str, tool_context: ToolContext
    ) -> dict:
    """Generates an ice-cream related image (an ice cream place or ice cream 
       itself) based on the prompt.

    Args:
        prompt (str): prompt for getting an image of ice cream or ice cream place

    Returns:
        dict: dict with image generation status, detail, and filename
    """

    response = client.models.generate_images(
        model=IMAGEN_MODEL_ID,
        prompt=prompt,
        config=GenerateImagesConfig(
            number_of_images = 1,
            person_generation = "DONT_ALLOW",
            safety_filter_level = "BLOCK_MEDIUM_AND_ABOVE"
            )
      )

    if not response.generated_images:
        return {'status': 'failed'}

    image_bytes = response.generated_images[0].image.image_bytes

    await tool_context.save_artifact(
      'image.png',
      Part.from_bytes(data=image_bytes, mime_type='image/png')
    )

    return {
      'status': 'success',
      'detail': 'Image generated successfully and stored in artifacts.',
      'filename': 'image.png'
    }


ice_cream_agent = Agent(
    name="ice_cream_agent",
    model=GEMINI_MODEL_ID,
    description=("""
        Ice cream agent to find ice cream places across the US, get information
        about them from Google Maps and Alok's personal reviews and ratings, and
        generate images of ice cream or ice cream places.
        """
    ),
    instruction=("""
        You are an ice cream agent that can find ice cream places across the US
        and relevant information about those places from Google Maps and Alok's
        personal reviews and ratings. You can also generate images of ice cream
        or ice cream places. Greet the user at the beginning the conversation by
        explaining what you do.
        
        Choose the appropriate tool to respond to queries about ice cream
        places, trying to leverage Alok's personal info when it's relevant to a
        particular response. If there's a more general query about ice cream
        place user ratings, distances, or ones that are currently open, favor
        the Google Maps tool. Use both places tools when responding to queries
        where it makes sense to have both the "big-picture" public perspective
        from Google Maps as well as Alok's personal curated experience (e.g. get
        top ice cream places in a particular location from both Maps and Alok's
        personal history).

        For any queries involving generating ice cream images, use the image
        generation tool. When generating an image in reference to something
        about a specific real-world ice cream place, make sure to note the image
        isn't actually from that place, but rather generated using AI.
    """),
    tools=[
        get_aloks_ice_cream_places_info,
        get_google_maps_ice_cream_places_info,
        generate_ice_cream_image
        ]
)

root_agent = ice_cream_agent