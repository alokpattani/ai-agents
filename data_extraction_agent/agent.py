from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext

import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

import base64

import json
import numpy as np
import pandas as pd

load_dotenv()

GOOGLE_GENAI_USE_VERTEXAI = os.getenv('GOOGLE_GENAI_USE_VERTEXAI')
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_CLOUD_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION')
GEMINI_MODEL_ID = os.getenv('GEMINI_MODEL_ID')

genai_client = genai.Client(vertexai=GOOGLE_GENAI_USE_VERTEXAI,
    project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)


def read_uploaded_file(tool_context: ToolContext):
    """
    Reads a PDF or text file uploaded by the user and processes it

    Args:
        tool_context: context object provided by ADK framework containing user content

    Returns:
        inline PDF in "data" form
    """
    user_content = tool_context.user_content
    
    if not user_content or not user_content.parts:
        return "No file was uploaded. Please upload a PDF file to analyze."

    # Loop through each Part in the 'parts' list to find uploaded content
    for i, part in enumerate(user_content.parts):
        print(f"Processing part {i}: {part}")

        # Check for inline data (uploaded files)
        if hasattr(part, "inline_data") and part.inline_data:
            mime_type = part.inline_data.mime_type
            data = part.inline_data.data

            if not mime_type or not data:
                continue
    
            try:
                # Handle PDF files only
                if mime_type == "application/pdf":
                    return data

                else:
                    return f"Unsupported file type: {mime_type}. Please upload a PDF."

            except Exception as e:
                return f"Error processing file: {str(e)}"

    return ("No readable file content found. Please upload a PDF. "
        f"Here is the tool context: {tool_context.user_content}")


async def save_structured_response_as_csv(
    structured_response: str,
    filename: str,
    tool_context: ToolContext
) -> str:
    """
    Saves structured response (e.g. from Gemini w/ controlled generation) as CSV
    as artifact in tool_context

    Args:
        structured_response: response from Gemini in text 
        filename: The name of the CSV file to save (without ".csv" in name)
        tool_context: context object provided by ADK framework containing user content

    Returns:
        A success message and the filename of the saved CSV.
    """

    df = pd.DataFrame(json.loads(structured_response))

    csv_data = df.to_csv(index=False)

    # Save the CSV content as an artifact
    version = await tool_context.save_artifact(
        filename=f"{filename}.csv", 
        artifact=types.Part.from_text(text=csv_data)
        )

    file_name_with_version = f"{filename}_{version}.csv"

    # Save CSV off locally since sometimes artifacts pane doesn't work in adk web
    df.to_csv(file_name_with_version, index=False)

    return f"Successfully saved structured data to artifact {file_name_with_version}."


async def generate_skating_data_from_pdf(tool_context: ToolContext):
    """Returns figure skating data with specified schema

    Args:
        tool_context: The context object provided by the ADK framework containing user content.

    Returns:
        pd.DataFrame: pandas dataframe with extracted figure skating data
    """

    pdf_data = read_uploaded_file(tool_context)

    document = types.Part.from_bytes(
        data=pdf_data,
        mime_type="application/pdf"
    )

    text = types.Part.from_text(text=
        """Extract all data from the included PDF into the following schema:
        Rank
        NOC
        LastName
        FirstName
        Gender
        Competition
        Location
        Date
        Program (Short or Free)
        ElementNumber
        ElementCode (e.g., 4F)
        BaseValue
        GOE
        ScoresofPanel (e.g., )

        Remove any non-traditional characters (e.g. trademarks, etc.) from all
        fields.
        There should 1 record for each element for each skater for each program.
        """)

    contents = [
        types.Content(
        role="user",
        parts=[text,
            document
            ]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 0,
        top_p = 1,
        seed = 0,
        max_output_tokens = 65535,
        response_mime_type = "application/json",
        response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
            "Rank": {
                "type": "STRING",
                "description":
                "The skater's overall rank in the competition."
            },
            "NOC": {
                "type": "STRING",
                "description":
                "The National Olympic Committee (nation) of the skater."
            },
            "LastName": {
                "type": "STRING",
                "description": "The last name of the skater."
            },
            "FirstName": {
                "type": "STRING",
                "description": "The first name of the skater."
            },
            "Gender": {
                "type": "STRING",
                "description": "The gender of the skater.",
                "enum": [
                "Male",
                "Female",
                "Pair"
                ]
            },
            "Competition": {
                "type": "STRING",
                "description": "The name of the competition."
            },
            "Location": {
                "type": "STRING",
                "description": "The location where the competition took place."
            },
            "Date": {
                "type": "STRING",
                "format": "date",
                "description": "The date of the program (YYYY-MM-DD)."
            },
            "Program": {
                "type": "STRING",
                "description": "The type of program.",
                "enum": [
                "Short",
                "Free"
                ]
            },
            "ElementNumber": {
                "type": "INTEGER",
                "description":
                "The sequential number of the element within the program."
            },
            "ElementCode": {
                "type": "STRING",
                "description":
                "The code for the element (e.g., 4F for quadruple flip)."
            },
            "BaseValue": {
                "type": "NUMBER",
                "format": "float",
                "description": "The base value of the element."
            },
            "GOE": {
                "type": "NUMBER",
                "format": "float",
                "description": "Grade of Execution (GOE) score for the element."
            },
            "ScoresofPanel": {
                "type": "NUMBER",
                "format": "float",
                "description": "Scores of panel for the element (e.g. 4.00)"
            },
            },
            "required": [
            "Rank",
            "NOC",
            "LastName",
            "FirstName",
            "Gender",
            "Competition",
            "Location",
            "Date",
            "Program",
            "ElementNumber",
            "ElementCode",
            "BaseValue",
            "GOE",
            "ScoresofPanel"
            ]
        },
        "description":
            "A list of individual element scores, each representing a single "
            "element from a specific skater's program."
        },
        thinking_config=types.ThinkingConfig(
        thinking_budget=128,
        )
    )

    response = genai_client.models.generate_content(
        model = GEMINI_MODEL_ID,
        contents = contents,
        config = generate_content_config,
        )

    response_text = response.text.replace('\n', ' ')   

    save_csv_result = await save_structured_response_as_csv(
        structured_response=response_text,
        filename="skating_data",
        tool_context=tool_context
        )

    return save_csv_result


async def generate_aerials_data_from_pdf(tool_context: ToolContext):
    """Returns ski aerials data with specified schema

    Args:
        tool_context: context object provided by ADK framework containing user content

    Returns:
        pd.DataFrame: pandas dataframe with extracted ski aerials data
    """

    pdf_data = read_uploaded_file(tool_context)

    document = types.Part.from_bytes(
        data=pdf_data,
        mime_type="application/pdf"
    )

    text = types.Part.from_text(text=
        """Extract all data from the included PDF into the following schema:
        NOC (e.g. CHN, CAN, USA)
        FISCode
        LastName
        FirstName
        Gender
        Competition
        Event
        Location
        Date
        Round
        JumpNumber (for given athlete in a given event/round)
        JumpCode
        Score

        Remove any non-traditional characters (e.g. trademarks, etc.) from all
        fields.
        There should be 1 record for each jump for each athlete for each
        competition.
        """)

    contents = [
        types.Content(
        role="user",
        parts=[text,
            document
            ]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 0,
        top_p = 1,
        seed = 0,
        max_output_tokens = 65535,
        response_mime_type = "application/json",
        response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
            "NOC": {
                "type": "STRING",
                "description":
                "The National Olympic Committee (nation) of the athlete."
            },
            "FISCode": {
                "type": "INTEGER",
                "description": "FIS code of the athlete."
            },
            "LastName": {
                "type": "STRING",
                "description": "The last name of the athlete."
            },
            "FirstName": {
                "type": "STRING",
                "description": "The first name of the athlete."
            },
            "Gender": {
                "type": "STRING",
                "description": "The gender of the athlete",
                "enum": [
                "Male",
                "Female"
                ]
            },
            "Competition": {
                "type": "STRING",
                "description": "The name of the overall competition."
            },
            "Event": {
                "type": "STRING",
                "description":
                "The name of the specific event within the competition (e.g. "
                "Men's Aerials)."
            },
            "Location": {
                "type": "STRING",
                "description":
                "The location where the competition/event took place."
            },
            "Date": {
                "type": "STRING",
                "format": "date",
                "description": "The date of the competition (YYYY-MM-DD)."
            },
            "Round": {
                "type": "STRING",
                "description": "The round within the competition.",
                "enum": [
                "Final",
                "Final 1",
                "Final 2",
                "Qualification",
                "Qualification Run 1",
                "Qualification Run 2"
                ]
            },
            "JumpNumber": {
                "type": "INTEGER",
                "description":
                "The number of this jump for this athlete in this event/round "
                " (i.e. 1 for their 1st jump, 2 for their 2nd, etc.)"
            },
            "JumpCode": {
                "type": "STRING",
                "description": "The code for the jump (e.g. 'bFdFF')."
            },
            "Score": {
                "type": "NUMBER",
                "format": "float",
                "description": "The base value of the element."
            }
            },
            "required": [
            "NOC",
            "LastName",
            "FirstName",
            "Gender",
            "Competition",
            "Event",
            "Location",
            "Date",
            "Round",
            "JumpNumber",
            "JumpCode",
            "Score"
            ]
        },
        "description":
            "A list of individual jump scores, each representing a single jump "
            "from a specific athlete's program."
        },
        thinking_config=types.ThinkingConfig(
        thinking_budget=128,
        )
    )

    response = genai_client.models.generate_content(
        model = GEMINI_MODEL_ID,
        contents = contents,
        config = generate_content_config,
        )

    response_text = response.text.replace('\n', ' ')

    save_csv_result = await save_structured_response_as_csv(
        structured_response=response_text,
        filename="aerials_data",
        tool_context=tool_context
        )

    return save_csv_result


data_extraction_agent = Agent(
    name="data_extraction_agent",
    model=GEMINI_MODEL_ID,
    description="""
        Data extraction agent to extract data from different Olympic sports
        provided in PDF form
        """,
    instruction="""
        You are a data extraction agent that extracts data provided in PDF form
        from different Olympic sports into CSV files with a specified schema.
        
        When a PDF is uploaded, determine if the PDF fits 1 of the following sports:
        1) figure skating - if so, use the generate_skating_data_from_pdf tool
        2) ski aerials - if so, use the generate_ski_aerials_data_from_pdf tool

        Check carefully if the PDF fits 1 of these sports - if not, respond to
        to the user that this PDF doesn't meet the criteria for data extraction
        and don't use any of the tools.

        If 1 of the tools successfully extracts the relevant data, let the user 
        know the name of the resulting CSV file (including version). If there's
        an error in data extraction, let the user know what it was.
    """,
    output_key = "data_extraction_agent_output",
    tools=[
        generate_skating_data_from_pdf,
        generate_aerials_data_from_pdf
        ]
)

root_agent = data_extraction_agent