from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer
from flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRConfig
from torch_split.interface import SplitClient

# load models


class PreFLMRInterface(SplitClient):
    def __init__(self):
        super().__init__()
        checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-G"
        image_processor_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        # flmr_config = FLMRConfig.from_pretrained(checkpoint_path)

        query_tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, subfolder="query_tokenizer", trust_remote_code=True
        )
        context_tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, subfolder="context_tokenizer", trust_remote_code=True
        )

        self.model = FLMRModelForRetrieval.from_pretrained(
            checkpoint_path,
            query_tokenizer=query_tokenizer,
            context_tokenizer=context_tokenizer,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
        self.query_tokenizer = query_tokenizer
        self.context_tokenizer = context_tokenizer

    def get_query_tokens(self, text):
        return self.query_tokenizer(text, return_tensors="pt")

    def get_model(self):
        return self.model

    def batch_sizes(self) -> list[int]:
        return [1, 2, 4, 8, 16, 32]

    def get_benchmarks(self, batch_size: int):
        def get_example_inputs(bs: int):
            while True:
                img = [Image.new("RGB", (224, 224), color=(255, 255, 255)) for _ in range(bs)]
                enc = self.image_processor(img)
                yield (), enc

        return 10, 30, get_example_inputs(batch_size)


instance = PreFLMRInterface()
ret = instance.get_query_tokens("hello world")
print(ret)
print(type(ret))
