from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.tools import FunctionTool, ToolContext, load_artifacts

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
    Reads a PDF file uploaded by the user and returns its byte content.
    Raises ValueError if a valid PDF is not found.

    Args:
        tool_context: context object provided by ADK framework containing user 
          content

    Returns:
        inline PDF in "data" form
    """
    user_content = tool_context.user_content
    
    if not user_content or not user_content.parts:
        # return "No file was uploaded. Please upload a PDF file to analyze."
        raise ValueError(
            "No file was uploaded. Please upload a PDF file to analyze.")

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
                    # return (f"Unsupported file type: {mime_type}. "
                    #     "Please upload a PDF.")
                    raise ValueError("Unsupported file type found: "
                        f"'{mime_type}'. Please upload a PDF.")

            except Exception as e:
                return f"Error processing file: {str(e)}"

    # return ("No readable file content found. Please upload a PDF. "
    #     f"Here is the tool context: {tool_context.user_content}")

    raise ValueError(
        "No readable PDF file content found. Please upload a valid PDF.")



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
        tool_context: context object provided by ADK framework containing user
          content

    Returns:
        A success message and the filename of the CSV saved as an artifact
    """

    # Convert structured response to pandas df
    df = pd.DataFrame(json.loads(structured_response))

    # Convert df to CSV to bytes using UTF-8 encoding
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    # Save CSV content as an artifact from bytes
    version = await tool_context.save_artifact(
        filename=f"{filename}.csv", 
        artifact=types.Part.from_bytes(data = csv_bytes, mime_type = "text/csv")
    )

    # Save df off as local CSV since sometimes Artifacts pane doesn't show CSVs
    # in adk web
    # df.to_csv(f"{filename}_{version}.csv", index=False)

    return f"Saved data to artifact {filename}.csv with version {version}."


async def get_table_schema_from_pdf(tool_context: ToolContext) -> str:
    """Returns table schema from given PDF to be used in data extraction

    Args:
        tool_context: context object provided by ADK framework containing user
          content

    Returns:
        str: schema to be used in PDF data extraction
    """

    try:
        pdf_data = read_uploaded_file(tool_context)
    except ValueError as e:
        return str(e)

    document = types.Part.from_bytes(
        data=pdf_data,
        mime_type="application/pdf"
    )

    text = types.Part.from_text(text=
        """
        Looking closely at the tables or other such structured data in the PDF,
        create a JSON schema that would be appropriate for extracting that data
        from the PDF into a structured output format that could be turned into a
        CSV or data frame. Return that JSON schema by itself in a format that
        can be used in downstream data extraction.
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
        # response_schema = {},
        thinking_config=types.ThinkingConfig(
        thinking_budget=128,
        )
    )

    response = genai_client.models.generate_content(
        model = GEMINI_MODEL_ID,
        contents = contents,
        config = generate_content_config,
        )

    return response.text


async def generate_data_from_pdf_and_schema(schema: str, tool_context:
    ToolContext):
    """Extracts data from PDF with specified schema

    Args:
        schema: JSON schema for use in PDF data extraction
        tool_context: context object provided by ADK framework containing user
          content

    Returns:
        pd.DataFrame: pandas dataframe with extracted data
    """

    try:
        pdf_data = read_uploaded_file(tool_context)
    except ValueError as e:
        return str(e)

    document = types.Part.from_bytes(
        data=pdf_data,
        mime_type="application/pdf"
    )

    text = types.Part.from_text(text=
        f"""Extract all data from the included PDF into the following schema:
        {schema}
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
        # response_schema = {},
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
        filename="extracted_data",
        tool_context=tool_context
        )

    return save_csv_result


data_extraction_agent = Agent(
    name="data_extraction_agent",
    model=GEMINI_MODEL_ID,
    description="""
        Agent to extract data from provided PDF into structured format
        """,
    instruction="""
        You are a data extraction agent that extracts data provided in PDF form
        into a CSV file with an appopriate schema.
        
        When a PDF is uploaded, generate the appropriate JSON schema for the 
        data in the PDF using the get_table_schema_from_pdf tool. Display that
        generated schema to the user and allow them to verify that it makes 
        sense. If the user suggest changes, pass those instructions along as
        context to the get_table_schema_from_pdf tool for further calls as
        necessary, checking in with the user after each change.

        If there is no structured data at all in the PDF contents, return a
        message to the user noting that, and potentially ask them to upload
        another PDF that has more structured data.

        Once the schema has been agreed upon, call the 
        generate_data_from_pdf_and_schema tool with that schema and the original
        PDF to extract the data into the desired form and export to CSV.

        Once the relevant data has been extracted, let the user know the name of
        the resulting CSV file (including version) and that it should be
        available in the 'Artifacts' pane in adk web. If there's an error in
        data extraction or CSV creation, let the user know what it was.
    """,
    output_key = "data_extraction_agent_output",
    tools=[
        get_table_schema_from_pdf,
        generate_data_from_pdf_and_schema,
        load_artifacts
        ]
)

root_agent = data_extraction_agent

app = App(
    name='data_extraction_agent_app',
    root_agent=root_agent,
    plugins=[SaveFilesAsArtifactsPlugin()],
)