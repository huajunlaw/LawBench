from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from loguru import logger

# Specify paths and hyperparameters for quantization
model_path = "models/old_model"
quant_path = "models/old_model_quant"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load your tokenizer and model with AutoAWQ
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)

dataset = [[
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me who you are."},
    {"role": "assistant", "content": "I am a large language model named Qwen..."}
]]
data = []
for msg in dataset:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    logger.info(text)
    data.append(text.strip())
logger.info(data)
model.quantize(tokenizer, quant_config=quant_config)

model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
tokenizer.save_pretrained(quant_path)
