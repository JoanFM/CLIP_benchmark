from transformers import AutoModel, AutoTokenizer
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

MODEL_NAME = 'jinaai/jina-embeddings-v2-base-en'
BLIP2_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
PROMPT = 'Describe this image in detail.'


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class JinaModel:

    def __init__(self, device='cpu', *args, **kwargs):
        self.device = device
        self.text_encoder = AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self.text_encoder.to(device)
        self.text_encoder.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.processor = Blip2Processor.from_pretrained(BLIP2_MODEL_NAME)
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            BLIP2_MODEL_NAME, torch_dtype=torch.float16
        )
        self.blip_model.to(device)
        self.blip_model.eval()

    def eval(self):
        self.text_encoder.eval()
        self.blip_model.eval()

    def encode_text(self, batch_tokens):
        batch_encoded_input = self.tokenizer.pad(
            {'input_ids': batch_tokens['input_ids']},
            return_tensors="pt",
        ).to(self.device)
        batch_model_output = self.text_encoder(**batch_encoded_input)
        embeddings = mean_pooling(
            batch_model_output, batch_encoded_input["attention_mask"]
        )
        return embeddings

    def encode_image(self, batch_images):
        for k in batch_images.keys():
            batch_images[k] = batch_images[k].squeeze(1).to(self.device)
        generated_ids = self.blip_model.generate(**batch_images, max_new_tokens=8192)
        generated_text = [text.strip() for text in self.processor.batch_decode(generated_ids, skip_special_tokens=True)]
        embeddings = self.text_encoder.encode(generated_text, convert_to_tensor=True)
        return embeddings


def load_jina_almost_clip(device="cpu", *args, **kwargs):
    """return transform: torchvision transform applied to images,
        model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`"""

    model = JinaModel(device=device)

    def transform(*args, **kwargs):
        return model.processor(text=PROMPT, return_tensors="pt", *args, **kwargs)

    return model, transform, model.tokenizer
