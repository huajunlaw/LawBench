import asyncio
import json
import re

import aiohttp
from loguru import logger

timeout = aiohttp.ClientTimeout(total=600)  # 设置超时时间


def read_json(input_file):
    with open(input_file, encoding="utf-8") as f:
        data_list = json.load(f)
    return data_list


def replace_tag_content(content, tag):
    pattern = fr'(<{tag}>).*?(</{tag}>)'
    replacement = r''
    return re.sub(pattern, replacement, content, flags=re.DOTALL)


async def _exec_single_query(cnt, item, predictions, endpoint="http://127.0.0.1:11434", api_key="xxx", model_name="", params: dict = {}, enable_think=False, is_rag=False):
    origin_prompt = f"{item['instruction']}\n{item['question']}"
    if is_rag:
        prompt = f"{item['rag_instruction'][:10000]}\n{item['question']}"
    else:
        prompt = origin_prompt
    messages = [{"role": "system", "content": "你是一名法律专家，给你一些参考内容，你可以作为参考，如果没有你需要的信息，你可以自行回答。你只需回答问题，不要做任何解释和说明。"}, {"role": "user", "content": prompt}]
    if enable_think:
        # temperature=0.6, top_p=0.95, top_k=20
        req_json = {"messages": messages, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 8192, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": True}}
    else:
        prompt += "/no_think"
        req_json = {"messages": messages, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 8192, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": False}}
    if model_name:
        req_json['model'] = model_name
    if params and isinstance(params, str):
        req_json.update(json.loads(params))
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{endpoint}/v1/chat/completions", json=req_json, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout) as response:
            resp = await response.json()
            prediction = resp['choices'][0]['message']["content"] or resp['choices'][0]['message']["reasoning_content"] or ""
            prediction = replace_tag_content(prediction, 'think').replace("<></>", "").strip()
            predictions[f"{cnt}"] = {
                    "origin_prompt": origin_prompt,
                    "real_prompt": prompt,
                    "prediction": prediction,
                    "refr": item["answer"],
                }
            logger.info(req_json)
            logger.info(prediction)


async def query_complitions(endpoint, api_key, model_name, params, output_file, data_list, think=False, rag=True):
    predictions = {}
    tasks = []
    for cnt, item in enumerate(data_list):
        try:
            task = asyncio.create_task(_exec_single_query(cnt, item, predictions, endpoint=endpoint, api_key=api_key, model_name=model_name, params=params, enable_think=think, is_rag=rag))
        except Exception as E:
            logger.info(E)
            continue
        tasks.append(task)
    await asyncio.gather(*tasks, return_exceptions=False)
    with open(output_file, "w") as f:
        f.write(json.dumps(predictions, ensure_ascii=False))
