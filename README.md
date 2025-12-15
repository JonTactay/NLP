# Dự án cuối kỳ NLP HK1 25–26 — Fine-tune & Đánh giá Transformer (Việt)

Repo này tổng hợp 3 hướng tiếp cận Transformer:
- **Encoder-only**: phân loại văn bản (PhoBERT / ViSoBERT)
- **Encoder-decoder**: sinh văn bản có input rõ ràng (ViT5 – tóm tắt)
- **Decoder-only**: sinh văn bản tự do / language modeling (Qwen3-0.6B – VietNews)

Ngoài fine-tuning, repo còn có:
- **Data Augmentation** bằng **synthetic data** sinh từ LLM
- **Đánh giá truyền thống** (Accuracy/F1, ROUGE/BLEU/BERTScore, Perplexity…)
- **LLM-based Evaluation** (Gemini/OpenAI chấm điểm chất lượng đầu ra)

> Lưu ý: Repo chỉ public **code**. Dataset/checkpoint không push lên GitHub.

---

## 1) Files trong repo

- `Bert_EncoderOnly.ipynb`  
  Fine-tune **PhoBERT / ViSoBERT** cho bài toán **toxicity/hate speech classification** (3 lớp).
  Có pipeline:
  - baseline data
  - data augmentation (synthetic)
  - preprocessing (emoji/teencode)
  - training + compare (Original vs Augmented)
  - test + demo/API
  - audit chất lượng synthetic

- `ViT5_encoderdecoder.ipynb`  
  Fine-tune **ViT5 (encoder-decoder)** cho **tóm tắt tiếng Việt** (seq2seq).
  Có:
  - load dataset (HF)
  - fine-tune + inference
  - đánh giá ROUGE/BLEU/BERTScore
  - LLM-based eval (Gemini/OpenAI)
  - phần bổ sung: **Synthetic Data từ nhiều LLM** (paraphrase summary) + đánh giá chất lượng synthetic

- `Qwen3_DecoderOnly.ipynb`  
  Fine-tune **Qwen/Qwen3-0.6B (decoder-only)** theo hướng **Causal LM** trên dữ liệu tin tức Việt (VietNews).
  Có:
  - build field text từ `title` + `content`
  - tokenize + train
  - đánh giá **loss/perplexity**
  - sinh văn bản tiếng Việt từ prompt
  - LLM-based eval (Gemini chấm 4 tiêu chí)
  - tạo synthetic bằng paraphrase/viết lại tin tức

---

## 2) Cài đặt môi trường

Khuyến nghị chạy trên **Google Colab** (GPU).

### Cài thư viện
```bash
pip install -U transformers datasets accelerate sentencepiece evaluate sacrebleu rouge-score bert-score scikit-learn
pip install -U underthesea py-vncorenlp emoji
pip install -U google-generativeai openai
