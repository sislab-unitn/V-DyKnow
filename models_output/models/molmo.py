class GenerationCollator:
    def __init__(
        self,
        processor,
        instruction,
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):

        samples = []
        questions = []
        images = []
        sample_ids = []

        for sample_id, question, image in batch:
            # samples.append(f"{self.instruction} {question}")
            samples.append(f"{self.instruction} {question}" if self.instruction else question)
            questions.append(question)
            if image is not None:
                images.append(image)
            sample_ids.append(sample_id)

        assert len(images) == 0 or len(images) == len(sample_ids)

        if len(images) == 0:
            images = None

        # MOLMo for now cannot process multiple samples at once, just pop the first item since we will use it with batch_size = 1
        input_ids = self.processor.process(
            text=samples.pop(), images=images, return_tensors="pt", padding=True
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
