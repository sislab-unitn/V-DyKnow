import base64
import json
import os
from typing import Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from utils import get_questions

SAMPLE_IDS_SEP="__"

EXP_WITH_IMG = ["visual", "detection"]
EXP_NO_IMG = ["text_only"]

def encode_image_b64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class DyKnowDataset(Dataset):
    def __init__(self, data_folder: str, experiment: str, b64_encode: bool = False):
        self.sample_ids = []
        self.questions = []
        self.images = []
        self.experiment = experiment
        self.b64_encode = b64_encode

        img_folder = os.path.join(data_folder, "imgs", "resized")

        with open(os.path.join(data_folder, "annotations", "wikidata_combined.json"), "r") as f:
            entries = json.load(f)

        # duplicate questions for images: e.g., one for flag, one for coat_of_arms
        for category, category_questions in tqdm(entries.items(), desc="Categories"):
            for subject, relations in tqdm(category_questions.items(), desc=f"Loading questions for {category}"):
                for relation, questions in relations.items():
                    if relation == "images":
                        continue
                    if self.experiment == "detection":
                        relation = "no_rel"
                    for q_type, q in questions[f"{self.experiment}_questions"].items():

                        if self.experiment in EXP_WITH_IMG:
                            img_urls = relations["images"]
                        elif self.experiment in EXP_NO_IMG:
                            img_urls = {"no_image": None}
                        else:
                            raise Exception(f"experiment '{experiment}' not supported.")

                        for img_type, img_url in img_urls.items():
                            assert SAMPLE_IDS_SEP not in category
                            assert SAMPLE_IDS_SEP not in subject
                            assert SAMPLE_IDS_SEP not in q_type
                            assert SAMPLE_IDS_SEP not in img_type

                            sample_id = SAMPLE_IDS_SEP.join([category, subject, relation, q_type, img_type])

                            img_path = None
                            if img_url is not None:
                                img_file = img_url.split("/")[-1]
                                img_name = ".".join(img_file.split(".")[:-1])
                                img_path = os.path.join(img_folder, img_file)
                                if img_file.endswith(".svg"):
                                    img_path = os.path.join(img_folder, f"{img_name}.png")
                                try:
                                    assert os.path.exists(img_path)
                                except:
                                    print(f"Path '{img_path}' does not exist for {sample_id}")
                                    continue

                            q_format = q
                            if self.experiment == "detection" and q_type == "rephrased":
                                if img_type == "coat_of_arms":
                                    q_format = q.format("coat of arms")
                                else:
                                    q_format = q.format(img_type)

                            self.sample_ids.append(sample_id)
                            self.questions.append(q_format)
                            self.images.append(img_path)

                    if self.experiment == "detection":
                        # skip relations
                        break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[str, str, Optional[Image.Image]]:
        img = None
        if self.experiment in EXP_WITH_IMG:
            if self.b64_encode:
                type_image = self.images[idx].split(".")[-1]
                img = f"data:image/{type_image};base64,{encode_image_b64(self.images[idx])}"
            else:
                img = Image.open(self.images[idx])
        return (
            self.sample_ids[idx],
            self.questions[idx],
            img
        )
