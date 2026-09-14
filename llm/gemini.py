from google import genai
from config import GEMINI_API_KEY
from tenacity import retry, stop_after_attempt, wait_exponential


client = genai.Client(api_key=GEMINI_API_KEY)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=20))
def generate_gemini_response(prompt):

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text
