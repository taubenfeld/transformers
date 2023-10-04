import torch

# from transformers import LlamaModel, LlamaConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

def slice_pkv(pkv, slice):
    """ 
    Each layer contains groups of different embedding types (e.g. key, value)
    Each embedding has dimensions [batch, heads, seq_len, embed_dim / heads]
    """
    out = []
    for layer in pkv:
        out.append([group[:, :, slice, :] for group in layer])
    return out

model_weights = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"

tokenizer = AutoTokenizer.from_pretrained(model_weights)
model = LlamaForCausalLM.from_pretrained(
                model_weights, device_map="auto", quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))

generation_config = GenerationConfig(
    temperature=0,
    top_p=0.75,
    # num_beams=1,
    use_cache=True,
    output_hidden_states=True,
    return_dict_in_generate=True,
    # output_past_key_values=True,
)

prompt = "Hey, are you conscious?"
input_ids = tokenizer.encode(
    prompt, return_tensors="pt").to(0) #.cuda()

generation_output = model.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    max_new_tokens=10,
    past_key_values=None,
)

# The previous cache, should be at length -1 of the inputs
# we will be passing to generate. Because next word prediction
# models don't have embeddings for the last token.
past_key_values_len = input_ids.shape[-1] - 1
pkv = generation_output.past_key_values
pkv = slice_pkv(pkv, slice=slice(0, past_key_values_len))

generation_output = model.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    max_new_tokens=10,
    past_key_values=pkv,
)

print(tokenizer.decode(generation_output.sequences[0]))
