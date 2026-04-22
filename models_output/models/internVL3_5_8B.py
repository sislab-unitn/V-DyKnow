
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
        conversations = []
        sample_ids = []

        for sample_id, question, image in batch:
            questions.append(question)
            sample_ids.append(sample_id)

            content = [{
                "type": "text",
                "text": (
                    f"{self.instruction} {question}"
                    if self.instruction
                    else question
                ),
            }]
            if image is not None:
                content.insert(0, {"type": "image", "image": image})

            conversation = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            conversations.append(conversation)
        
        input_ids = self.processor.apply_chat_template(
                conversations, 
                padding=True, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
                )
        
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

            conversation = [
                {
                    "role": "user",
                    "content": f"{self.instruction} {question}" if self.instruction else question
                }
            ]

            conversations.append(conversation)

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for msg in conversations
        ]
        input_ids = self.processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        )

        return input_ids, questions, sample_ids
