
from transformers import AutoConfig, AutoModelForCausalLM



# Load only the configuration without downloading weights
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Qwen 1.5B model
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

# Create model with random weights (no pretrained weights downloaded)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# Print the model architecture
print(model)

# Print parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
