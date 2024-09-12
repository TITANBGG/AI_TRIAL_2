import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# GPU'nun mevcut olup olmadığını kontrol edelim (sorun yaratır mı araştır)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GPT-2 tokenizer'ı yükleyelim 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 için EOS token'ını pad token olarak kullanıyoruz.

# IMDb veri setini yükleyelim 
dataset = load_dataset("imdb")

# Tokenizasyon fonksiyonu
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # labels olarak input_ids'leri kullanıyoruz
    return tokens

# IMDb veri setini tokenize edelim
tokenized_imdb = dataset['train'].map(tokenize_function, batched=True)

# GPT-2 modelini ve tokenizer'ını yükleyelim
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.to(device)  # Modeli GPU'ya taşıyoruz



"""  dfvgcxvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"""

# Modeli eğitme için ayarları yapalım
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Batch boyutunu ayarlıyoruz
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer ile modeli eğitelim
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb,
)

trainer.train()

# Modeli kaydettim
model.save_pretrained("./gpt2_imdb_model")
tokenizer.save_pretrained("./gpt2_imdb_model")


def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Kullanıcıdan bir soru alalım
user_input = input("Soru: ")
response = generate_response(user_input)
print(f"Roger AI: {response}")

