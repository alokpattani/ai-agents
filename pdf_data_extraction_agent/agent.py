from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
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

async def get_pdf_from_artifact(
    filename: str, 
    tool_context: ToolContext
    ) -> bytes:
    """
    Loads a specific PDF from saved artifacts and returns its byte content

    Args:
        filename: Name of PDF artifact file to load
        tool_context: context object provided by ADK framework

    Returns:
        Byte content (bytes) of PDF file

    Raises:
        ValueError: If the artifact is not found or is not a PDF
        RuntimeError: For unexpected storage or other errors
    """
    try:
        # Load the specified artifact
        pdf_artifact = await tool_context.load_artifact(filename=filename)

        # Check if the artifact was found and contains data
        if (pdf_artifact and hasattr(pdf_artifact, 'inline_data') and 
            pdf_artifact.inline_data):
            # Validate that it is a PDF
            if pdf_artifact.inline_data.mime_type == "application/pdf":
                print(f"✅ Successfully loaded PDF artifact '{filename}'.")
                # Extract and eturn the raw byte content
                pdf_bytes = pdf_artifact.inline_data.data
                return pdf_bytes
            else:
                # Raise an error if the file type is wrong
                raise ValueError(
                    f"Artifact '{filename}' is not a PDF. "
                    f"Found type: '{pdf_artifact.inline_data.mime_type}'."
                )
        else:
            # Raise an error if the artifact wasn't found or was empty
            raise ValueError(f"Artifact '{filename}' not found or is empty.")

    except ValueError as e:
        # This will catch errors from load_artifact or the checks above
        print(f"❌ Error loading artifact: {e}")
        raise e
    except Exception as e:
        # Handle other potential storage or unexpected errors
        raise RuntimeError(f"An unexpected error occurred: {e}")

async def get_table_schema_from_pdf(
    filename: str,
    tool_context: ToolContext
    ) -> str:
    """Returns table schema from given PDF to be used in data extraction

    Args:
        tool_context: context object provided by ADK framework

    Returns:
        str: schema to be used in PDF data extraction
    """

    try:
        pdf_data = await get_pdf_from_artifact(filename, tool_context)
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
        CSV or data frame. Use minimal nesting within such a schema so that the
        output can be more easily turned into tabular format (where reasonable),
        even if this involves repeating various dimension fields in the output.
        
        Return that JSON schema by itself in a format that can be used in
        downstream data extraction.
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

    return f"Saved data to artifact {filename}.csv with version {version}."


async def generate_data_from_pdf_and_schema(
    filename: str,
    schema: str, 
    tool_context: ToolContext
    ):
    """Extracts data from PDF with specified schema

    Args:
        schema: JSON schema for use in PDF data extraction
        tool_context: context object provided by ADK framework containing user
          content

    Returns:
        pd.DataFrame: pandas dataframe with extracted data
    """

    try:
        pdf_data = await get_pdf_from_artifact(filename, tool_context)
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


pdf_data_extraction_agent = Agent(
    name="pdf_data_extraction_agent",
    model=GEMINI_MODEL_ID,
    description="""
        Agent to extract data from provided PDF into structured format
        """,
    instruction="""
        You are a data extraction agent that extracts data provided in PDF form
        into a CSV file with an appopriate schema.
        
        When a PDF is uploaded, generate the appropriate JSON schema for the 
        data in the PDF using the get_table_schema_from_pdf tool, taking into
        account any special instructions provided by the user. Display that
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
        generate_data_from_pdf_and_schema
        ]
)

root_agent = pdf_data_extraction_agent

app = App(
    name='pdf_data_extraction_agent',
    root_agent=root_agent,
    plugins=[SaveFilesAsArtifactsPlugin()],
)