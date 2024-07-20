from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from curlora import *
from lora import *
import evaluate
import gc


mrpc_dataset = load_dataset("glue", "mrpc")
wikidataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
sst_dataset = load_dataset("glue", "sst2")
sentiment_dataset = load_dataset("sentiment140")

txt = wikidataset["text"]
txt = [s for s in txt if s != '']
txt = "".join(txt)

device = "cuda"
max_len = 512 # 256 for GPT2 and RoBERTa
lr = 2.5e-4


def calculate_perplexity(model, tokenizer, text, device='cuda'):
    if not text.strip():
        print("Warning: Empty text encountered")
        return float('inf')
    
    encodings = tokenizer(text, return_tensors='pt', truncation=True, padding = True, max_length=10000) # only max_len for GPT2 and RoBERTa
    input_ids = encodings.input_ids.to(device)
    
    if input_ids.numel() == 0:
        print(f"Warning: Empty input_ids for text: {text[:100]}...")
        return float('inf')
    
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        
    loss = outputs.loss
    perplexity = torch.exp(loss)
    
    try:
        return perplexity.item()
    finally:
        input_ids.detach().cpu()
        target_ids.detach().cpu()
        outputs.logits.detach().cpu()
        torch.cuda.empty_cache()
        _ = gc.collect()


# I used this function "replace_linear_with_lora" from
# https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-E/01_main-chapter-code/appendix-E.ipynb
def replace_linear_with_curlora(model, rank, alpha):
    for name, module in model.named_children():
        #if isinstance(module, torch.nn.Linear):
        if any(l in name for l in ["q_proj", "v_proj", "k_proj"]):
            setattr(model, name, LinearWithCURLoRA(module, rank, alpha))
        else:
            replace_linear_with_curlora(module, rank, alpha)


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        #if isinstance(module, torch.nn.Linear):
        if any(l in name for l in ["q_proj", "v_proj", "k_proj"]):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)


def load_model_and_tokenizer(model_name, type = "lora"):
    #model = AutoModelForSequenceClassification.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    for param in model.parameters():
        param.requires_grad = False

    if type == "lora":
        replace_linear_with_lora(model, rank=16, alpha=1)
    else:
        replace_linear_with_curlora(model, rank=16, alpha=1)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters after: {total_params:,}")

    model.to(device)
    
    ppl = calculate_perplexity(model, tokenizer, txt)
    print("Perplexity:", round(ppl, 2))
    
    torch.manual_seed(1311)
    num_classes = 2
    lm_head = model.lm_head
    
    model.lm_head = torch.nn.Linear(in_features=4096, out_features=num_classes)
    model.to(device)
    
    return model, tokenizer, lm_head.to(device)

def evaluate_sst2(model, tokenizer, dataset, device):

    dataset = dataset["validation"]
    model.eval()
    #model.to(device)
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset):
        inputs = tokenizer.encode(example["sentence"], return_tensors="pt",
                                  truncation=True, padding=True, max_length = max_len).to(device)
        with torch.no_grad():
            #outputs = model(**inputs)
            outputs = model(inputs)#["logits"][:, -1, :]
        
        predicted = torch.argmax(outputs.logits[:, -1, :]).item()
        #predicted = (outputs >= 0.5).item()
        correct += (predicted == example["label"])
        total += 1
    
    accuracy = correct / total
    return accuracy


def evaluate_mrpc(model, tokenizer, dataset, device):
    dataset = dataset["validation"]
    
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset):
        inputs = tokenizer.encode(example["sentence1"], example["sentence2"], return_tensors="pt",
                                  truncation=True, padding=True, max_length = max_len).to(device)
        with torch.no_grad():
            #outputs = model(**inputs)
            outputs = model(inputs)#["logits"][:, -1, :]
        
        predicted = torch.argmax(outputs.logits[:, -1, :]).item()
        #predicted = (outputs >= 0.5).item()
        correct += (predicted == example["label"])
        total += 1
    
    accuracy = correct / total
    return accuracy


def evaluate_sentiment(model, tokenizer, dataset, device):

    dataset = dataset["train"]
    model.eval()
    #model.to(device)
    
    correct = 0
    total = 0
    batch_size = 128
    
    for i in tqdm(range(0, 1000, batch_size)):
        example = dataset[i:i+batch_size]
        inputs = tokenizer(example["text"], return_tensors="pt",
                           truncation=True, padding = True, max_length = max_len).to(device)
        with torch.no_grad():
            #outputs = model(**inputs)
            outputs = model(**inputs)#["logits"][:, -1, :]
        
        predicted = torch.argmax(outputs.logits[:, -1, :], dim = 1).detach().cpu()#.item()
        #predicted = (outputs >= 0.5).item()
        correct += (predicted == (torch.Tensor(example["sentiment"]) // 4)).sum().item()
        total += len(example["text"])
    
    accuracy = correct / total
    return accuracy
