import argparse
import ast
import csv
import json
import os
from typing import List 

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_topic_tag(raw: str) -> List[str]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        inside = s[1:-1].strip()
        if not inside:
            return []
        parts = inside.split(",")
        return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]

    parts = s.split(",")
    return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]


def safe_str(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def build_text(row: pd.Series) -> str:
    zh_none = "\u65E0"
    zh_unknown = "\u672A\u77E5"
    zh_comma = "\uFF0C"

    caption = safe_str(row.get("caption", ""))
    cover = safe_str(row.get("manual_cover_text", ""))

    tags = parse_topic_tag(row.get("topic_tag", ""))
    tags_text = zh_comma.join(tags) if tags else zh_none

    c1 = safe_str(row.get("first_level_category_name", ""))
    c2 = safe_str(row.get("second_level_category_name", ""))
    c3 = safe_str(row.get("third_level_category_name", ""))

    cat_chain = " > ".join([x for x in [c1, c2, c3] if x])
    if not cat_chain:
        cat_chain = zh_unknown

    text = (
        f"\u89C6\u9891\u6807\u9898\uFF1A{caption if caption else zh_none}\\n"
        f"\u5C01\u9762\u6587\u5B57\uFF1A{cover if cover else zh_none}\\n"
        f"\u8BDD\u9898\u6807\u7B7E\uFF1A{tags_text}\\n"
        f"\u89C6\u9891\u5206\u7C7B\uFF1A{cat_chain}"
    )
    return text


def load_encoder(model_name: str, use_fp16: bool, local_files_only: bool = False):
    last_error = None

    try:
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

        def encode_fn(batch_texts: List[str], batch_size: int, max_length: int):
            out = model.encode(
                batch_texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return np.asarray(out["dense_vecs"], dtype=np.float32)

        backend = "FlagEmbedding.BGEM3FlagModel"
        return encode_fn, backend
    except Exception as e:
        last_error = e

    # Fallback to pure transformers (PyTorch) to avoid tf/keras dependency issues.
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as e:
        hint = (
            "Model load failed. This is usually network timeout to huggingface.co, not CSV format. "
            "Try one of: "
            "(1) --local_model_dir /path/to/local/bge-m3, "
            "(2) --hf_endpoint https://hf-mirror.com, "
            "(3) pre-download model then use --local_files_only."
        )
        raise RuntimeError(
            f"{hint}\n"
            f"FlagEmbedding error: {repr(last_error)}\n"
            f"Transformers error: {repr(e)}"
        ) from e

    model.to(device)
    model.eval()

    if use_fp16 and device == "cuda":
        model.half()

    def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_fn(batch_texts: List[str], batch_size: int, max_length: int):
        embs = []
        with torch.no_grad():
            for i in range(0, len(batch_texts), batch_size):
                sub = batch_texts[i : i + batch_size]
                batch_inputs = tokenizer(
                    sub,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                out = model(**batch_inputs)
                pooled = mean_pooling(out.last_hidden_state, batch_inputs["attention_mask"])
                embs.append(pooled.detach().cpu().numpy())

        return np.asarray(np.concatenate(embs, axis=0), dtype=np.float32)

    backend = "transformers.AutoModel(mean_pooling)"
    return encode_fn, backend


def load_csv_robust(path: str, required_cols: List[str]) -> pd.DataFrame:
    # Try fast C engine first.
    try:
        return pd.read_csv(path, usecols=required_cols, low_memory=False)
    except Exception as e:
        print(f"[Load] C-engine read failed: {type(e).__name__}: {e}")

    # Fall back to Python engine and skip malformed lines.
    try:
        return pd.read_csv(
            path,
            usecols=required_cols,
            engine="python",
            on_bad_lines="skip",
            quoting=csv.QUOTE_MINIMAL,
            encoding="utf-8",
            encoding_errors="replace",
        )
    except Exception as e:
        print(f"[Load] Python-engine read failed: {type(e).__name__}: {e}")

    # Last fallback: read all columns with max tolerance, then select required cols.
    df_all = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip",
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        encoding_errors="replace",
    )
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing required columns after robust read: {missing}")
    return df_all[required_cols].copy()


def run(args):
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    os.makedirs(args.output_dir, exist_ok=True)

    required_cols = [
        "video_id",
        "manual_cover_text",
        "caption",
        "topic_tag",
        "first_level_category_name",
        "second_level_category_name",
        "third_level_category_name",
    ]

    print(f"[Load] Reading: {args.input_csv}")
    total_rows = 0
    with open(args.input_csv, "r", encoding="utf-8", errors="replace") as f:
        total_rows = max(0, sum(1 for _ in f) - 1)

    df = load_csv_robust(args.input_csv, required_cols)
    if total_rows > 0 and len(df) < total_rows:
        print(f"[Load] Parsed rows: {len(df)} / {total_rows}. Potential malformed rows skipped: {total_rows - len(df)}")

    df = df.dropna(subset=["video_id"]).copy()
    df["video_id"] = pd.to_numeric(df["video_id"], errors="coerce").fillna(-1).astype(np.int64)
    df = df[df["video_id"] >= 0].copy()

    print(f"[Build] Building text prompts for {len(df)} rows...")
    texts = [build_text(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Prompt")]
    video_ids = df["video_id"].to_numpy(dtype=np.int64)

    model_ref = args.local_model_dir if args.local_model_dir else args.model_name
    print(f"[Model] Loading {model_ref} ...")
    encode_fn, backend = load_encoder(model_ref, args.use_fp16, local_files_only=args.local_files_only)
    print(f"[Model] Backend: {backend}")

    all_embs = []
    n = len(texts)
    for start in tqdm(range(0, n, args.batch_size), desc="Encode"):
        end = min(start + args.batch_size, n)
        batch_texts = texts[start:end]
        embs = encode_fn(batch_texts, batch_size=args.batch_size, max_length=args.max_length)
        all_embs.append(embs)

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)

    if args.l2_normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms

    print(f"[Save] shape={embeddings.shape}, dtype={embeddings.dtype}")
    video_path = os.path.join(args.output_dir, "video_ids.npy")
    emb_path = os.path.join(args.output_dir, "bge_m3_embeddings.npy")
    np.save(video_path, video_ids)
    np.save(emb_path, embeddings)

    meta = {
        "model_name": args.model_name,
        "backend": backend,
        "input_csv": args.input_csv,
        "num_rows": int(len(video_ids)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else -1,
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "l2_normalize": bool(args.l2_normalize),
        "use_fp16": bool(args.use_fp16),
        "video_ids_file": os.path.basename(video_path),
        "embeddings_file": os.path.basename(emb_path),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Done] Saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/data/cbn01/KuaiRec/data/kuairec_caption_category.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/cbn01/KuaiRec/data/bge_m3_caption_embeddings",
    )
    parser.add_argument("--model_name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--local_model_dir", type=str, default="", help="Optional local directory of bge-m3 model")
    parser.add_argument("--local_files_only", action="store_true", default=False, help="Force loading model from local files only")
    parser.add_argument("--hf_endpoint", type=str, default="", help="Optional HuggingFace mirror endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_fp16", action="store_true", default=True)
    parser.add_argument("--l2_normalize", action="store_true", default=False)

    run(parser.parse_args())
