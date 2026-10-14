import json, os, re
from openai import OpenAI

KEYS = {"OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL", "tabpfn_key"}
for line in open("../.env"):
    k, _, v = line.partition("=")
    if k.strip() in KEYS:
        os.environ["TABPFN_TOKEN" if k.strip() == "tabpfn_key" else k.strip()] = v.strip().strip("'\"")

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
usage = {"calls": 0, "in_tokens": 0, "out_tokens": 0}


def ask(system, user):
    r = client.chat.completions.create(model=os.environ["MODEL"], temperature=0,
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    usage["calls"] += 1
    usage["in_tokens"] += r.usage.prompt_tokens
    usage["out_tokens"] += r.usage.completion_tokens
    return json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group(0))
