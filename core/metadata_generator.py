import google.generativeai as genai
import json
from utils.logger import logger
import config

class MetadataGenerator:
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        if not self.api_key:
            logger.warning("Gemini API Key not found. Metadata generation will fail.")
        else:
            genai.configure(api_key=self.api_key)
            # User requested 'gemini-2.5-flash-lite', falling back to stable 'gemini-1.5-flash' if needed
            self.models_to_try = [
                'gemini-2.5-flash-lite', 
                'gemini-1.5-flash', 
                'gemini-2.0-flash-lite-preview-02-05',
                'gemini-pro'
            ]

    def generate_metadata(self, transcript):
        """
        Generate viral metadata (Title, Description, Hashtags) from transcript.
        """
        if not self.api_key:
            return {"error": "API Key missing"}

        prompt = config.METADATA_PROMPT_TEMPLATE.replace("{transcript}", transcript)
        last_error = None

        for model_name in self.models_to_try:
            try:
                logger.info(f"Attempting metadata generation with model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                
                # Extract JSON from response
                text_response = response.text
                
                # Clean up markdown code blocks if present
                if "```json" in text_response:
                    text_response = text_response.split("```json")[1].split("```")[0].strip()
                elif "```" in text_response:
                    text_response = text_response.split("```")[1].strip()
                    
                data = json.loads(text_response)
                
                # If we got here, success!
                data['generated_by'] = model_name
                return data
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                last_error = e
                # Continue to next model
        
        # If all failed
        logger.error(f"All metadata models failed. Last error: {last_error}")
        return {"error": f"All models failed. Last error: {str(last_error)}"}

if __name__ == "__main__":
    # Test
    generator = MetadataGenerator()
    sample_transcript = "This is a video about how to make the best coffee at home. You just need beans and water."
    print(generator.generate_metadata(sample_transcript))
