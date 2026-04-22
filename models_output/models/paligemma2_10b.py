from PIL import Image

class GenerationCollator:
    def __init__(
        self,
        processor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):
        questions = []
        texts = []
        images = []
        sample_ids = []

        for sample_id, question, image in batch:
            questions.append(question)
            sample_ids.append(sample_id)
            prompt = ""
            # https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions?hl=it
            if image is not None:
                prompt = f"<image>answer en {self.instruction} {question}\n" if self.instruction else f"<image>answer en {question}\n"
            else:
                prompt = f"<image>{self.instruction} {question}\n" if self.instruction else f"<image>{question}\n"
                # if no image is provided, use a blank image
                image = Image.new("RGB", (672, 672), color=(0, 0, 0))

            
            texts.append(prompt)
            images.append(image)
        
        input_ids = self.processor(text=texts, images=images, padding=True, return_tensors="pt")

        return input_ids, questions, sample_ids


class LLMGenerationCollator:
    def __init__(
        self,
        processor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):
        questions = []
        conversations = []
        texts = []
        sample_ids = []

        for sample_id, question, image in batch:
            questions.append(question)
            sample_ids.append(sample_id)
            # https://huggingface.co/google/gemma-2-9b-it
            conversation = [
                {
                    "role": "user",
                    "content": f"{self.instruction} {question}" if self.instruction else question
                }
            ]

            conversations.append(conversation)

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in conversations
        ]
        input_ids = self.processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        )


        return input_ids, questions, sample_ids
