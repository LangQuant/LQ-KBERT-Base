# LQ-KBERT-Base: Crypto Market Korean Sentiment & Action Signal Classifier

[LangQuant](https://langquant.com)에서 공개한 **한국어 금융 커뮤니티/뉴스 투자심리 분류 모델**입니다.  
`klue/bert-base`를 백본으로 하고, 가상자산 관련 한국어 데이터셋 **10만 건 이상**을 전처리하여 파인튜닝했습니다.  
모델은 문장 단위 입력(`≤200자`)에 대해 **투자 심리·행동·감정·확신도·관련성·유해성**을 동시에 예측합니다.

- [Github](https://github.com/LangQuant/LQ-KBERT-Base)
- [Huggingface](https://huggingface.co/langquant/LQ-Kbert-base)
---
### 모델의 아웃풋은 다음과 같습니다.

```json
{
  "sentiment_strength": "strong_pos | weak_pos | neutral | weak_neg | strong_neg",
  "action_signal": "buy | hold | sell | avoid | info_only | ask_info",
  "emotions": ["greed","fear","confidence","doubt","anger","hope","sarcasm"],
  "certainty": 0.0 ~ 1.0,
  "relevance": 0.0 ~ 1.0,
  "toxicity": 0.0 ~ 1.0
}
```

---
## 사용법
```
import torch, json
from transformers import AutoTokenizer, AutoModel

repo_or_dir = "LangQuant/LQ-Kbert-base" 
texts = [
    "비트코인 조정 후 반등, 투자심리 개선",
    "환율 급등에 증시 변동성 확대",
    "비트 그만 좀 내려라 진짜..",
    "폭락ㅠㅠㅜㅠㅜ 다 팔아야할까요?"
]


tokenizer = AutoTokenizer.from_pretrained(repo_or_dir)
model = AutoModel.from_pretrained(repo_or_dir, trust_remote_code=True)
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

```

---
## Labeling Guidelines

### Sentiment Strength
- **strong_pos**: 급등 확신, `"가즈아"`, `"무조건 간다"`.
- **weak_pos**: 조심스러운 낙관, `"반등 가능"`, `"괜찮을 듯"`.
- **neutral**: 단순 정보/공지/잡담.
- **weak_neg**: 완곡한 부정, `"조정 올 듯"`, `"관망"`.
- **strong_neg**: 폭락·패닉, `"나락"`, `"망함"`, `"해킹/제재"`.

### Action Signal
- **buy**: 매수/진입 지시, `"지금 산다"`, `"롱"`.
- **hold**: 보유 유지/관망, `"존버"`, `"유지"`.
- **sell**: 매도/청산, `"익절"`, `"손절"`, `"정리"`.
- **avoid**: 회피/위험 경고, `"가지마"`, `"스캠"`, `"위험"`.
- **info_only**: 단순 정보 전달 (뉴스/공지).
- **ask_info**: 질문/탐색, `"들어가도 돼?"`, `"왜 떨어져?"`.

### Emotions (다중 선택)
- **greed** 탐욕  
- **fear** 두려움  
- **confidence** 확신  
- **doubt** 의심  
- **anger** 분노  
- **hope** 희망  
- **sarcasm** 풍자  

### Certainty
- **0.2~0.4**: 질문·탐색·밈 (낮음)  
- **0.4~0.6**: 완곡한 의견 (중간)  
- **0.6~0.8**: 수치·근거·공식성 (높음)  
- **0.8~1.0**: 강한 단정·지시 (매우 높음)  

### Relevance
- **0.7~1.0**: 직접적인 투자/시장 관련  
- **0.4~0.7**: 간접 관련 (업계/인물/기술)  
- **0.0~0.3**: 무관/잡담/밈  

### Toxicity
- 욕설·모욕·비하 강도에 따라 **0~1**.  
- 투자 의미와는 별도로 독립적으로 평가.  

---

## Sentiment Strength vs Action Signal

- **Sentiment Strength**  
  - 투자 심리의 강도 (긍정 ↔ 부정).  
  - 가격 전망의 톤에 집중.  

- **Action Signal**  
  - 실제 투자 행동 의도/지시.  
  - 매수/매도/보유/회피/질문/정보.  


---

### 예시

| 문장 | sentiment_strength | action_signal | 해석 |
|------|--------------------|---------------|------|
| "개떡상이여 " | strong_pos | buy | 강한 상승 확신 + 즉시 매수 의도 |
| "여기선 관망이 맞다" | weak_neg | hold | 부정적이지만 보유 유지 선택 |
| "들어가도 될까?" | weak_pos | ask_info | 조심스러운 낙관, 매수 탐색 질문 |
| "해킹 터짐, 비상. 접근 금지" | strong_neg | avoid | 강한 부정 + 회피 권고 |
| "업데이트 공지 나왔습니다" | neutral | info_only | 단순 정보 제공, 행동 없음 |

---
### Citation
```
@misc{langquant2025lkbert,
  title  = {LQ-KBERT-Base: Crypto Market Korean Sentiment & Action Signal Classifier},
  author = {LangQuant},
  year   = {2025},
  url    = {https://huggingface.co/langquant/LQ-Kbert-base}
}
```
---
### Disclaimer
```
이 모델은 학술 연구 및 실험용으로만 제공됩니다.
본 모델의 출력은 금융/투자 자문으로 간주될 수 없으며,
발생하는 모든 결과에 대해 LangQuant는 책임을 지지 않습니다.
```
