import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. Chargement de l'environnement quantifié
model_name = "Qwen/Qwen2.5-Coder-1.5B"
adapter_path = "lora_atlas_model_vanilla"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

print("Chargement du modèle de base...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)

print("Greffe de l'adaptateur ATLAS...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# 2. Préparation du Prompt
# ATTENTION : Il faut utiliser EXACTEMENT la même structure ChatML que pendant l'entraînement.
instruction = "Generate a Lego assembly sequence."
prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 3. Génération
print("Génération de la séquence en cours...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,  # Longueur suffisante pour quelques briques
        temperature=0.2,  # Température basse = on limite l'hallucination, on veut la syntaxe pure
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

# 4. Extraction et affichage du résultat
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
# On isole uniquement la réponse de l'assistant
result = (
    generated_text.split("<|im_start|>assistant\n")[-1]
    .replace("<|im_end|>", "")
    .strip()
)

print("\n=== Séquence Générée par ATLAS ===")
print(result)
print("==================================")
