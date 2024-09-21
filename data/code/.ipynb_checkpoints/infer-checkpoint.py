import torch
def predict_glm4(prompt,model,tokenizer):
    
   
    inputs = tokenizer.apply_chat_template([{"role": "system", "content": "您将担任评审的专家角色"},{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to('cuda')


    gen_kwargs = {"max_length": 2048, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def predict_qwen2(prompt,model,tokenizer):
    
    messages = [
    {"role": "system", "content": "您将担任评审的专家角色"},
    {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def predict_internlm2_5(prompt,model,tokenizer):
    
    inputs = tokenizer.apply_chat_template([{"role": "system", "content": "您将担任评审的专家角色"},{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to('cuda')


    gen_kwargs = {"max_length": 2048, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def predict_gemma2(prompt,model,tokenizer):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True).to('cuda')
    gen_kwargs = {"max_length": 2048, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
