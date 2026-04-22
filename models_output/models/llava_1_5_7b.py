
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
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding=True,
                return_tensors="pt",
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
            # Reference: https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md
            conversation = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: {self.instruction} {question}\nASSISTANT:"
            conversations.append(conversation)

        input_ids = self.processor(
            text=conversations,
            padding=True,
            return_tensors="pt",
        )

        return input_ids, questions, sample_ids
