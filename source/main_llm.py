from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def load_model(
    name: str, torch_dtype: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch_dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def query(
    prompt: list[dict[str, str]],
    model_name: str,
    torch_dtype: str,
    max_new_tokens: int,
) -> str:
    model, tokenizer = load_model(model_name, torch_dtype)
    input_tokens = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([input_tokens], return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generated_ids = model.generate(
        **model_inputs, max_new_tokens=max_new_tokens, streamer=streamer
    )
    generated_text = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_text, skip_special_tokens=True)[0]
