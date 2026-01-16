from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
print("Tokenizer loaded.")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
print("Model loaded successfully.")
