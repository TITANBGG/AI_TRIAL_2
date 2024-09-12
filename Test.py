import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# 1. GPU'nun mevcut olup olmadığını kontrol edelim
# Eğer GPU varsa, modeli ve veriyi GPU'ya taşıyacağız. Aksi takdirde CPU kullanılır.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. GPT-2 tokenizer'ı yükleyelim
# Tokenizer, metin verisini modelin anlayabileceği token'lara (sayı dizilerine) dönüştürür.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Eğitilen modelin yolunu kullanmak istersen: "C:/Users/alinebierreset/Desktop/AI_2/gpt2_imdb_model"
tokenizer.pad_token = tokenizer.eos_token  # EOS (End of Sentence) token'ını pad token olarak kullanıyoruz.

# 3. IMDb veri setini yükleyelim
# Hugging Face'in 'datasets' modülünü kullanarak IMDb veri setini yükleriz. Bu veri seti, incelemeleri ve onların pozitif/negatif etiketlerini içerir.
dataset = load_dataset("imdb")

# 4. Tokenizasyon fonksiyonu
# Bu fonksiyon, IMDb veri setindeki metinleri GPT-2'nin anlayabileceği token dizilerine çevirir.
# Her bir inceleme metni max_length uzunluğuna kadar padding (boşluk) ile doldurulur veya kısaltılır.
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # labels olarak input_ids'leri kullanıyoruz (dil modelleme için).
    return tokens

# 5. IMDb veri setini tokenize edelim
# Tokenizasyon fonksiyonunu tüm veri setine uygulayarak metin verisini model için uygun hale getiriyoruz.
tokenized_imdb = dataset['train'].map(tokenize_function, batched=True)

# 6. GPT-2 modelini yükleyelim
# Hugging Face'den önceden eğitilmiş GPT-2 modelini yükleriz. Eğer kendi eğittiğin modeli kullanmak istersen, model yolunu değiştirebilirsin.
model = GPT2LMHeadModel.from_pretrained("gpt2")  # Eğitilen modelin yolunu kullanmak istersen: "C:/Users/alinebierreset/Desktop/AI_2/gpt2_imdb_model"
model = model.to(device)  # Modeli GPU'ya taşıyoruz (mevcutsa).

# 7. Modeli eğitme için ayarları yapalım
# Bu ayarlar modelin nasıl eğitileceğini belirler. Her 'epoch' (tüm veri seti üzerinde bir tam geçiş), modelin değerlendirilmesi yapılacak.
training_args = TrainingArguments(
    output_dir="./results",             # Eğitim sonuçlarının kaydedileceği dizin
    evaluation_strategy="epoch",        # Her epoch'tan sonra değerlendirme yapılacak
    num_train_epochs=1,                 # Model kaç kez eğitilecek (1 epoch)
    per_device_train_batch_size=4,      # Her cihazda kaç örnekle eğitilecek
    save_steps=10_000,                  # Model her 10.000 adımda bir kaydedilecek
    save_total_limit=2,                 # En fazla 2 model kaydı tutulacak
)

# 8. Trainer ile modeli eğitelim
# Hugging Face Trainer API'si, modeli kolayca eğitmemizi sağlar.
trainer = Trainer(
    model=model,                        # Eğitilecek model
    args=training_args,                 # Eğitim argümanları
    train_dataset=tokenized_imdb,       # Eğitilecek veri seti
)

trainer.train()  # Model eğitilir.

# 9. Modeli kaydedelim
# Eğitilen modeli ve tokenizer'ı gelecekte kullanmak üzere kaydediyoruz.
model.save_pretrained("./gpt2_imdb_model")  # Eğitilen model kaydedilir
tokenizer.save_pretrained("./gpt2_imdb_model")  # Tokenizer da kaydedilir

# 10. Soru-cevap fonksiyonu
# Bu fonksiyon, kullanıcıdan alınan soruyu GPT-2 modeline verip yanıt üretir.
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Kullanıcının sorusu tokenleştirilir ve cihaza (GPU/CPU) taşınır
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)  # Model yanıtı üretir
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  # Yanıt geri döndürülür (özel tokenlar çıkarılır)

# 11. Kullanıcıdan soru alalım
# Kullanıcı sürekli soru sorabilir, "exit" yazarak programdan çıkabilir.
while True:
    user_input = input("Soru (çıkmak için 'exit' yazın): ")
    if user_input.lower() == "exit":  # Kullanıcı 'exit' yazarsa döngüden çıkar.
        break
    response = generate_response(user_input)  # Kullanıcının sorusu yanıtlanır
    print(f"Roger AI: {response}")  # Yanıt terminale yazdırılır
