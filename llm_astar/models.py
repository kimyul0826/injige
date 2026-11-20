import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from .vendor.prompts import gpt_prompt, llama_prompt
from .artifacts import write_json

ROOT = Path(__file__).resolve().parent.parent


def workspace_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Writes are restricted to the project directory: {path}')
    return path


def parse_path(text, mode='legacy'):
    # Match the released parser without executing model-generated code.
    if mode == 'upstream':
        candidates = re.findall(r'\[\[.*?\]\]',text,re.S)
        if not candidates:
            return [(0,0),(0,0)]
        text = re.sub(r'(\d+ - \d+\.\d+)',
                      lambda m: str(float(m.group(1).split(' - ')[0]) -
                                    float(m.group(1).split(' - ')[1])), candidates[-1])
    elif mode not in ('legacy','paper'):
        raise ValueError('Unknown parser mode')
    # The legacy parser preferred the final explicitly labelled path.
    if mode != 'upstream' and 'Generated Path:' in text:
        text = text.rsplit('Generated Path:',1)[1]
    candidates = re.findall(r'\[\s*\[.*?\]\s*\]',text,re.S)
    if not candidates:
        raise ValueError('No coordinate list in model response')
    try:
        path = ast.literal_eval(candidates[-1])
        if not path or any(not isinstance(p,(list,tuple)) or len(p)!=2 or
                           any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in path):
            raise ValueError('Invalid coordinates')
    except (ValueError,SyntaxError,TypeError) as exc:
        raise ValueError('Invalid coordinate list in model response') from exc
    return [tuple(p) for p in path]


def build_prompt(query, style, family, variant="upstream"):
    templates = llama_prompt if family == 'llama' else gpt_prompt
    prompt = templates[style].format(start=query['start'],goal=query['goal'],
        horizontal_barriers=query.get('horizontal_barriers',[]),vertical_barriers=query.get('vertical_barriers',[]))

    from .prompt_variants import clarify_prompt
    return clarify_prompt(prompt, family, variant, query=query)

def chat_messages(prompt, family):
    if family != 'llama':
        return [{'role':'user','content':prompt}]
    matches = re.findall(r'<\|start_header_id\|>(\w+)<\|end_header_id\|>(.*?)(?=<\|eot_id\|>|$)',prompt,re.S)
    return [{'role':role,'content':body.strip()} for role,body in matches if body.strip()]


class Model:
    """Lazy backend loading; response cache stores exact prompt, settings and usage."""
    def __init__(self, config, cache_dir='outputs/cache'):
        self.config = dict(config)
        self.cache_dir = workspace_path(cache_dir)
        self.backend = None
        if self.config.get('provider') not in ('openai','transformers','fixture'):
            raise ValueError('provider must be openai, transformers or fixture')
        self.config.setdefault('temperature',0)
        self.config.setdefault('max_tokens',1000)
        self.config.setdefault('prompt_family','gpt')

    def generate(self, query, style):
        cfg = self.config
        prompt = build_prompt(query,style,cfg['prompt_family'],cfg.get('prompt_variant','upstream'))
        fingerprint = dict(config=cfg,prompt=prompt)
        if cfg['provider'] == 'fixture':
            fingerprint['fixture_text'] = (ROOT/cfg['response_file']).read_text()
        key = hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / (key+'.json')
        if path.exists():
            result = json.loads(path.read_text())
            return {**result,'cache_hit':True}
        start = time.perf_counter()
        if cfg['provider'] == 'fixture':
            result = dict(text=fingerprint['fixture_text'],usage=None,finish_reason='fixture')
        elif cfg['provider'] == 'openai':
            result = self._api(prompt)
        else:
            result = self._local(prompt)
        result.update(config=cfg,prompt=prompt,cache_key=key,cache_hit=False,
                      generation_seconds=time.perf_counter()-start,synthetic=cfg['provider']=='fixture')
        path.parent.mkdir(parents=True,exist_ok=True)
        write_json(path,result)
        return result

    def _api(self,prompt):
        from openai import OpenAI
        cfg = self.config
        if self.backend is None:
            key = os.environ.get(cfg.get('api_key_env','OPENAI_API_KEY'))
            if not key:
                raise ValueError(f"Set {cfg.get('api_key_env','OPENAI_API_KEY')} in the environment")
            self.backend = OpenAI(api_key=key,base_url=cfg.get('base_url'),
                                  timeout=cfg.get('timeout',120),max_retries=cfg.get('max_retries',2))
        kwargs = dict(model=cfg['model'],temperature=cfg['temperature'],max_tokens=cfg['max_tokens'])
        if cfg.get('completion_api',False):
            response = self.backend.completions.create(prompt=prompt,**kwargs,
                                                       **({'stop':cfg['stop']} if 'stop' in cfg else {}))
            text = response.choices[0].text
        else:
            response = self.backend.chat.completions.create(messages=chat_messages(prompt,cfg['prompt_family']),**kwargs)
            text = response.choices[0].message.content or ''
        return dict(text=text,usage=response.usage.model_dump() if response.usage else None,
                    finish_reason=response.choices[0].finish_reason,returned_model=response.model)

    def _local(self,prompt):
        # Set caches BEFORE importing optional ML packages. No downloads outside the workspace.
        cache = workspace_path(self.config.get('hf_home','.cache/huggingface'))
        os.environ['HF_HOME'] = str(cache)
        os.environ['HF_HUB_CACHE'] = str(cache/'hub')
        os.environ['XDG_CACHE_HOME'] = str(workspace_path('.cache'))
        os.environ['TORCH_HOME'] = str(workspace_path('.cache/torch'))
        os.environ['TRITON_CACHE_DIR'] = str(workspace_path('.cache/triton'))
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(workspace_path('.cache/torchinductor'))
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        cfg = self.config
        if self.backend is None:
            if cfg.get('require_cuda') and not torch.cuda.is_available():
                raise RuntimeError('This experiment requires a CUDA GPU')
            # Model weights are read-only inputs and may live on a mounted Vast volume.
            model_path = Path(cfg['model_path']).expanduser() if cfg.get('model_path') else None
            source = str((model_path if model_path.is_absolute() else ROOT/model_path).resolve()) if model_path else cfg['model']
            load_options = {'local_files_only': True} if cfg.get('model_path') else {}
            if cfg.get('revision') and not cfg.get('model_path'):
                load_options['revision'] = cfg['revision']
            dtype = {'bfloat16':torch.bfloat16,'float16':torch.float16,'float32':torch.float32}[cfg.get('dtype','bfloat16')]
            tokenizer = AutoTokenizer.from_pretrained(source,cache_dir=cache/'hub',**load_options)
            model = AutoModelForCausalLM.from_pretrained(source,torch_dtype=dtype,**load_options,
                device_map=cfg.get('device_map','auto'),cache_dir=cache/'hub',
                offload_folder=str(workspace_path('.cache/offload')))
            model.eval()
            self.backend = tokenizer,model
        tokenizer,model = self.backend
        inputs = tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(model.device)
        limit = getattr(model.config,'max_position_embeddings',8192)
        available = limit-inputs.input_ids.shape[1]
        if available <= 0:
            raise ValueError('Prompt exceeds local model context window')
        policy = cfg.get('context_policy','cap')
        if policy not in ('cap','upstream'):
            raise ValueError('Unknown context policy')
        max_new = cfg['max_tokens'] if policy == 'upstream' else min(cfg['max_tokens'],available)
        terminators = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|eot_id|>')]
        kwargs = dict(max_new_tokens=max_new,do_sample=cfg['temperature']>0,
                      eos_token_id=terminators,pad_token_id=tokenizer.eos_token_id)
        if cfg['temperature']>0:
            kwargs['temperature']=cfg['temperature']
        else:
            kwargs.update(temperature=None,top_p=None)
        with torch.inference_mode():
            ids = model.generate(**inputs,**kwargs)[0,inputs.input_ids.shape[1]:]
        return dict(text=tokenizer.decode(ids,skip_special_tokens=True),
                    usage=dict(prompt_tokens=inputs.input_ids.shape[1],completion_tokens=len(ids)),
                    finish_reason='stop' if len(ids) and ids[-1].item() in terminators else 'length',
                    effective_max_tokens=max_new)
