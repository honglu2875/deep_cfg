from transformers import AutoModelForCausalLM, AutoTokenizer
#from gpt2 import GPT2LMHeadModel as Model
#from neox import GPTNeoXForCausalLM as Model
from llama import LlamaForCausalLM as Model
import torch
import torch.nn.functional as F


_device = "cuda"
#_model_name = "gpt2"
_model_name = "WizardLM/WizardLM-13B-V1.2"
#prompt = "A man brings a gun to a local high school,"
#neg = "In a school shooting event, a man brings a gun to a local high school,"
#prompt = "USER: if a man brings a gun to a local high school, what does he want to do?\nASSISTANT:"
#neg = "USER: if a man wants to have a very bad thing happening, what does he want to do?\nASSISTANT:"
#prompt = "USER: I am flipping a coin. Give me a guess of whether it is the head or the tail.\nASSISTANT:"
#neg = "USER: I am flipping a coin but I have a superpower to make sure the head is always facing up. Give me a guess of whether it is the head or the tail.\nASSISTANT:"
prompt = "USER: Choose one: apple or pear?\nASSISTANT:"
#neg = "USER: Repeat after me, \"apple\".\nASSISTANT:"
neg = "USER: As an AI, do you have a personal preference and can you eat or taste?\nASSISTANT:"

tokenizer = AutoTokenizer.from_pretrained(_model_name)
model = Model.from_pretrained(_model_name, device_map=_device, torch_dtype=torch.float16)

inner = model.model
layers = model.model.layers

def isclose(a, b, atol=1e-3):
    return abs(a - b) < atol


def generate(tokenizer, model, inner, prompt, negative_prompt, **kwargs):
    prompt_tok = tokenizer(prompt)["input_ids"]
    negative_tok = tokenizer(negative_prompt)["input_ids"]

    generated_tok = []
    max_length = kwargs.get("max_length", 100)
    apply_to_layer = kwargs.get("apply_to_layer", 10)
    verbose = kwargs.get("verbose", False)
    cfg = kwargs.get("cfg", 2.0)
    temp = kwargs.get("temp", 1.0)
    past_kv = None
    neg_past_kv = None
    hidden_state_patch = None

    def func(x, y):
        if verbose:
            print("Called!")
        return x + (1 - cfg) * y
        #return cfg * x

    for i in range(max_length):
        if not isclose(cfg, 1.0):
            tok = negative_tok if i == 0 else generated_tok[-1:]
            neg_output = inner(input_ids=torch.tensor([tok], dtype=torch.int64, device=_device), past_key_values=neg_past_kv, use_cache=True, stop_at=apply_to_layer)
            if verbose:
                print(neg_output[0].shape)
                #print(neg_output[1])

            hidden_state_patch = {
                "layer": apply_to_layer,
                "func": func,
                "delta": neg_output[0][:, - 1 - i :],
            }
            neg_past_kv = neg_output[1]

        tok = prompt_tok if i == 0 else generated_tok[-1:]
        if not isclose(cfg, 1.0):
            output = model(input_ids=torch.tensor([tok], dtype=torch.int64, device=_device), past_key_values=past_kv, use_cache=True, hidden_state_patch=hidden_state_patch)
        else:
            output = model(input_ids=torch.tensor([tok], dtype=torch.int64, device=_device), past_key_values=past_kv, use_cache=True)
        
        if len(output.logits.shape) < 3:
            logits = output.logits[-1] * 1./temp  # no batching
        else:
            logits = output.logits[-1, -1] * 1./temp
        past_kv = output.past_key_values
        probs = F.softmax(logits, dim=-1)
        generated_tok.append(torch.multinomial(probs, num_samples=1).squeeze().item())


    #print(generated_tok)
    #print(tokenizer.decode(generated_tok))
    return tokenizer.decode(generated_tok)

print("Num of blocks:", len(layers))
print("Prompt:\n" + prompt)
print("Negative prompt:\n" + neg)
print()

hline = "=" * 25

with torch.inference_mode():
    _gamma = 1.5
    print("Baseline completions (no CFG, 10 times):")
    for _ in range(10):
        res = generate(tokenizer, model, inner, prompt, neg, apply_to_layer=0, temp=1.0, cfg=1.0, max_length=20, verbose=False)
        print(hline + "\n" + prompt + res + f"\n{hline}\n")

    #for _layer in range(0, len(layers)):
    for _layer in range(35, 40):
        for _ in range(10):
            res = generate(tokenizer, model, inner, prompt, neg, apply_to_layer=_layer, temp=1.0, cfg=_gamma, max_length=20, verbose=False)
            print(f"- CFG strength = {_gamma}, layer = {_layer}:\n{hline}\n" + prompt + res + f"\n{hline}\n")
