# Dự án cuối kỳ NLP — HK1 (2025–2026)
Repo này tổng hợp 3 notebook fine-tune Transformer theo 3 kiến trúc phổ biến:
- **Encoder-only** (BERT-style): phân loại văn bản (PhoBERT/ViSoBERT) trên ViHSD
- **Encoder–Decoder** (seq2seq): tóm tắt tiếng Việt (ViT5)
- **Decoder-only** (Causal LM): language modeling/sinh văn bản (Qwen3)

> Repo public **chỉ code** (không public dataset/checkpoints). Xem mục “Dataset & Checkpoints”.
 
## Dataset (ViHSD)
Bài toán: Vietnamese Hate Speech Detection (3 nhãn: CLEAN / OFFENSIVE / HATE).

Nguồn dataset:
- HuggingFace: `sonlam1102/vihsd`. 

---

## 1) Nội dung repo

### `Bert_EncoderOnly.ipynb`
**Bài toán:** Vietnamese Hate Speech Detection (3 nhãn: CLEAN / OFFENSIVE / HATE).  
**Mô hình:** PhoBERT / ViSoBERT (encoder-only).

Pipeline chính (tóm tắt):
- Load + chuẩn hóa dữ liệu
- Tiền xử lý (emoji/teencode)
- Fine-tune & đánh giá (Accuracy/F1)
- (Tuỳ chọn) Synthetic data augmentation + so sánh kết quả

### `ViT5_encoderdecoder.ipynb`
**Bài toán:** Tóm tắt tiếng Việt (seq2seq).  
**Mô hình:** ViT5 (encoder-decoder).

Pipeline chính:
- Load dataset (HuggingFace hoặc dữ liệu riêng)
- Fine-tune + inference
- Đánh giá: ROUGE / BLEU / BERTScore
- (Tuỳ chọn) LLM-based evaluation & synthetic paraphrase

### `Qwen3_DecoderOnly.ipynb`
**Bài toán:** Causal Language Modeling / sinh văn bản.  
**Mô hình:** Qwen3-0.6B (decoder-only).

Pipeline chính:
- Chuẩn hóa field text (title + content nếu dùng news)
- Fine-tune causal LM
- Đánh giá: loss / perplexity
- Sinh văn bản theo prompt
- (Tuỳ chọn) LLM-based scoring & synthetic data

---

## 2) Cài đặt môi trường

```bash
pip install -U transformers datasets accelerate sentencepiece evaluate
pip install -U scikit-learn sacrebleu rouge-score bert-score
pip install -U google-generativeai openai
pip install -U underthesea emoji
