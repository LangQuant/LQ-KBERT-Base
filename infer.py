import os, json, argparse, glob
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

REPO_ID = "langquant/LQ-Kbert-base"
CKPT_RELPATH = "model/lq-kbert-base.pt"

SENTI_MAP = {'strong_pos':0,'weak_pos':1,'neutral':2,'weak_neg':3,'strong_neg':4}
ACT_MAP   = {'buy':0,'hold':1,'sell':2,'avoid':3,'info_only':4,'ask_info':5}
EMO_LIST  = ['greed','fear','confidence','doubt','anger','hope','sarcasm']
IDX2SENTI = {v:k for k,v in SENTI_MAP.items()}
IDX2ACT   = {v:k for k,v in ACT_MAP.items()}

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def sigmoid(x): return 1/(1+np.exp(-x))

class KbertMTL(nn.Module):
    def __init__(self, base_model: nn.Module, hidden: int = 768):
        super().__init__()
        self.bert = base_model
        self.head_senti = nn.Linear(hidden, 5)
        self.head_act   = nn.Linear(hidden, 6)
        self.head_emo   = nn.Linear(hidden, 7)
        self.head_reg   = nn.Linear(hidden, 3)
        self.has_token_type = getattr(self.bert.embeddings, "token_type_embeddings", None) is not None

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if self.has_token_type and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.bert(**kwargs)
        h = out.last_hidden_state[:, 0]  # [CLS]
        return {
            "logits_senti": self.head_senti(h),
            "logits_act":   self.head_act(h),
            "logits_emo":   self.head_emo(h),
            "pred_reg":     self.head_reg(h)
        }

def load_ckpt_from_hub() -> Tuple[dict, str]:
    path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_RELPATH)
    obj = torch.load(path, map_location="cpu")
    return obj, path

def build_model_and_tokenizer(ckpt_obj: dict, hidden: int = 768):
    model_name = ckpt_obj.get("model_name", "klue/bert-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModel.from_pretrained(model_name)
    model = KbertMTL(base_model=base, hidden=hidden)

    state_dict = ckpt_obj["state_dict"] if "state_dict" in ckpt_obj else ckpt_obj
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f"[Warn] Missing keys: {missing[:5]}{' ...' if len(missing)>5 else ''}")
    if unexpected: print(f"[Warn] Unexpected keys: {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")

    emo_thr = float(ckpt_obj.get("emo_threshold", 0.5))
    return model, tokenizer, model_name, emo_thr

def ensure_min_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Input must contain a 'text' column.")
    if "id" not in df.columns:
        df = df.copy(); df["id"] = [f"row-{i}" for i in range(len(df))]
    return df[["id","text"]].copy()

def load_inputs(args) -> pd.DataFrame:
    if args.text is not None:
        return pd.DataFrame([{"id":"sample-0", "text":args.text}])
    if args.input_csv:
        return ensure_min_cols(pd.read_csv(args.input_csv, engine="python"))
    if args.input_tsv:
        return ensure_min_cols(pd.read_csv(args.input_tsv, sep="\t", engine="python"))
    if args.input_dir:
        files = sorted(glob.glob(str(Path(args.input_dir)/"*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV found in: {args.input_dir}")
        dfs = [ensure_min_cols(pd.read_csv(p, engine="python")) for p in files]
        return pd.concat(dfs, 0, ignore_index=True).drop_duplicates(subset=["id","text"])
    raise ValueError("Provide --text or --input_csv / --input_tsv / --input_dir")

@torch.no_grad()
def run_inference(df: pd.DataFrame,
                  model: KbertMTL,
                  tokenizer: AutoTokenizer,
                  device: str,
                  max_len: int,
                  batch_size: int,
                  emo_threshold: float) -> List[Dict]:
    model.to(device).eval()
    out_list: List[Dict] = []
    n = len(df)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        ids   = df["id"].iloc[s:e].tolist()
        texts = df["text"].iloc[s:e].astype(str).tolist()

        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc["token_type_ids"].to(device) if "token_type_ids" in enc else None

        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        senti = out["logits_senti"].argmax(-1).cpu().numpy()
        act   = out["logits_act"].argmax(-1).cpu().numpy()
        emo_p = sigmoid(out["logits_emo"].cpu().numpy())
        reg   = out["pred_reg"].cpu().numpy()

        for i in range(len(ids)):
            emos = [EMO_LIST[j] for j, p in enumerate(emo_p[i]) if p >= emo_threshold]
            out_list.append({
                "id": ids[i],
                "text": texts[i],
                "pred_sentiment_strength": IDX2SENTI[int(senti[i])],
                "pred_action_signal":      IDX2ACT[int(act[i])],
                "pred_emotions":           emos,
                "pred_certainty":  float(np.clip(reg[i,0], 0, 1)),
                "pred_relevance":  float(np.clip(reg[i,1], 0, 1)),
                "pred_toxicity":   float(np.clip(reg[i,2], 0, 1)),
            })
    return out_list

def save_outputs(items: List[Dict], out_path: str, fmt: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if fmt == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        df = pd.DataFrame(items)
        df["pred_emotions"] = df["pred_emotions"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

def build_argparser():
    ap = argparse.ArgumentParser(description="LQ-KBERT Inference (fixed Hub ckpt)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="단일 문장 추론")
    g.add_argument("--input_csv", type=str, help="CSV 경로(컬럼: id,text)")
    g.add_argument("--input_tsv", type=str, help="TSV 경로(컬럼: id,text)")
    g.add_argument("--input_dir", type=str, help="CSV 다수 폴더")

    ap.add_argument("--max_len", type=int, default=200, help="입력 최대 토큰 길이(권장 200자)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (기본 자동)")
    ap.add_argument("--emo_threshold", type=float, default=None, help="감정 멀티라벨 임계치 (기본: ckpt값 또는 0.5)")

    ap.add_argument("--out", type=str, default="outputs/infer.jsonl")
    ap.add_argument("--fmt", type=str, choices=["jsonl","csv"], default="jsonl")

    ap.add_argument("--seed", type=int, default=42)
    return ap

def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = load_inputs(args)

    ckpt_obj, ckpt_path = load_ckpt_from_hub()

    model, tokenizer, used_model_name, emo_thr_ckpt = build_model_and_tokenizer(ckpt_obj, hidden=768)
    emo_thr = float(args.emo_threshold) if args.emo_threshold is not None else float(emo_thr_ckpt)

    preds = run_inference(
        df, model, tokenizer,
        device=device, max_len=args.max_len,
        batch_size=args.batch_size, emo_threshold=emo_thr
    )

    # 5) 저장
    save_outputs(preds, args.out, fmt=args.fmt)
    print(f"[OK] {len(preds)} preds → {args.out}")
    print(f" - repo_id: {REPO_ID}")
    print(f" - ckpt:    {ckpt_path}")
    print(f" - emo_thr: {emo_thr:.3f}")

if __name__ == "__main__":
    main()
