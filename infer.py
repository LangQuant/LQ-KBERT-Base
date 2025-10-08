import torch, json
from transformers import AutoTokenizer, AutoModel

repo_or_dir = "LangQuant/LQ-Kbert-base" 
texts = [
    "비트코인 조정 후 반등, 투자심리 개선",
    "환율 급등에 증시 변동성 확대",
    "비트 그만 좀 내려라 진짜..",
    "폭락ㅠㅠㅜㅠㅜ 다 팔아야할까요?"
]


tokenizer = AutoTokenizer.from_pretrained(repo_or_dir, local_files_only=True)
model = AutoModel.from_pretrained(repo_or_dir, trust_remote_code=True, local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()


enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
with torch.inference_mode():
    out = model(**enc)

IDX2SENTI = {0:"strong_pos",1:"weak_pos",2:"neutral",3:"weak_neg",4:"strong_neg"}
IDX2ACT   = {0:"buy",1:"hold",2:"sell",3:"avoid",4:"info_only",5:"ask_info"}
EMO_LIST  = ["greed","fear","confidence","doubt","anger","hope","sarcasm"]


for i, t in enumerate(texts):
    senti = int(out["logits_senti"][i].argmax().item())
    act   = int(out["logits_act"][i].argmax().item())
    emo_p = torch.sigmoid(out["logits_emo"][i]).tolist()
    reg   = torch.clamp(out["pred_reg"][i], 0, 1).tolist()
    emos = [EMO_LIST[j] for j,p in enumerate(emo_p) if p >= 0.5]

    result = {
        "text": t,
        "pred_sentiment_strength": IDX2SENTI[senti],
        "pred_action_signal":      IDX2ACT[act],
        "pred_emotions":           emos,
        "pred_certainty":  float(reg[0]),
        "pred_relevance":  float(reg[1]),
        "pred_toxicity":   float(reg[2]),
    }
    print(json.dumps(result, ensure_ascii=False))
