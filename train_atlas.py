import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- 1. Configuration Modèle ---
model_name = "Qwen/Qwen2.5-Coder-1.5B"
max_seq_length = 2048

print("Configuration de la quantification (4-bit)...")
# Utilisation de bitsandbytes pour réduire l'empreinte VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=(
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ),
)

print("Chargement du modèle de base et du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = (
    tokenizer.eos_token
)  # Nécessaire pour éviter les erreurs de padding

# Si l'argument bnb_config fait planter le script sous Windows,
# supprime-le et ajoute torch_dtype=torch.float16 à la place.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Accélération de la répartition GPU/CPU
)

# --- 2. Configuration LoRA (L'Adaptateur) ---
print("Préparation du graphe PyTorch pour le modèle quantifié...")
# Cette ligne est vitale : elle active le gradient checkpointing proprement pour QLoRA
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

print("Injection de l'adaptateur LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 3. Préparation du Dataset ---
prompt_template = """<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"""


def format_dataset(example):
    return {
        "text": prompt_template.format(
            instruction=example["instruction"], output=example["output"]
        )
    }


print("Chargement et formatage du dataset...")
dataset = load_dataset("json", data_files="atlas_dev_dataset.jsonl", split="train")
dataset = dataset.map(format_dataset)

# --- 4. Configuration de l'Entraînement ---
print("Initialisation du SFTTrainer...")
training_args = TrainingArguments(
    output_dir="outputs_atlas_vanilla",
    # 1. On divise la charge par passe
    per_device_train_batch_size=1,
    # 2. On compense pour garder un batch effectif équivalent
    gradient_accumulation_steps=8,
    # 3. LE SAUVEUR DE VRAM : On recalcule les activations au lieu de les stocker
    gradient_checkpointing=True,
    warmup_steps=50,
    max_steps=200,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    # 4. On utilise la version 8-bit de l'optimiseur (grâce à bitsandbytes)
    # avec pagination (décharge la mémoire vers la RAM classique si besoin)
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)

# --- 5. Lancement ---
print("Démarrage de l'entraînement...")
trainer.train()

print("Sauvegarde de l'adaptateur LoRA...")
trainer.model.save_pretrained("lora_atlas_model_vanilla")
tokenizer.save_pretrained("lora_atlas_model_vanilla")
print("Terminé !")
