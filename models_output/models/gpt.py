from openai import OpenAI

API_KEY = "YOUR_API_KEY_HERE" # Replace with your actual API key

class GPTModel:
    def __init__(self, model_name, temperature=0):
        '''Initialize the GPTModel with the model name and configuration.'''
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(api_key=API_KEY)
    
    def generate(self, content, max_new_tokens=15):
        '''Generate content using the Gemini model.'''

        output = self.client.responses.create(
            model=self.model_name, 
            input=content, 
            max_output_tokens=max_new_tokens,
            #temperature=self.temperature,
            #config=self.config
            )
        return output.output_text
    

class GenerationCollator:
    def __init__(
        self,
        instruction="",
    ):
        self.instruction = instruction

    def __call__(self, sample_id, question, image):
        # https://platform.openai.com/docs/guides/images-vision
        content = [{"role": "user", "content": None}]
        query = f"{self.instruction} {question}" if self.instruction else question

        if image:
            content[0]["content"] = [
                { "type": "input_text", "text": query },
                {
                    "type": "input_image",
                    "image_url": image, #image passed as base64 string
                },]
        else:
            content[0]["content"] = query

        return content, question, sample_id