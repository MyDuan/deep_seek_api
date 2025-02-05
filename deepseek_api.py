from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

app = FastAPI()

class QueryRequest(BaseModel):
    system: str
    user_input: str
    max_new_tokens: int = 200

def get_outputs(model, model_input, max_new_tokens=200):
    with torch.no_grad():
        output_ids = model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    return output_ids

@app.post("/query")
def query_model(request: QueryRequest):
    try:
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user_input}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_input = tokenizer(prompt, return_tensors="pt")

        outputs = get_outputs(model, model_input, max_new_tokens=request.max_new_tokens)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {"response": response[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))