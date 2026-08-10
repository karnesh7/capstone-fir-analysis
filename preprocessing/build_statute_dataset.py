#!/usr/bin/env python3
"""
Build complete statute dataset:
1. Extract IPC sections from src_dataset_files/ipc.pdf
2. Load BNS sections from output/bns_sections_extracted.json
3. Generate output/statute_chunks_complete.jsonl
4. Generate embeddings with all-MiniLM-L6-v2 into output/statute_vectors.jsonl
5. Deploy to Pinecone index 'statute-embeddings'
"""

import json
import re
import os
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
SRC_DIR = REPO_ROOT / "src_dataset_files"


def classify_offence_type(text: str, title: str = "") -> str:
    combined = (title + " " + text).lower()
    if any(k in combined for k in ["theft", "stolen", "extortion", "robbery", "dacoity", "misappropriation", "breach of trust", "cheating", "mischief", "trespass", "property"]):
        return "property"
    if any(k in combined for k in ["murder", "culpable homicide", "hurt", "grievous hurt", "assault", "kidnapping", "abduction", "wrongful restraint", "confinement", "death", "force"]):
        return "person"
    if any(k in combined for k in ["rape", "sexual", "modesty", "voyeurism", "stalking", "unnatural"]):
        return "sexual"
    if any(k in combined for k in ["unlawful assembly", "rioting", "affray", "public servant", "contempt", "public tranquility"]):
        return "public_order"
    if any(k in combined for k in ["forgery", "counterfeit", "coin", "stamp", "currency", "deception", "fraud"]):
        return "economic"
    return "other"


def extract_ipc_sections() -> dict[str, dict]:
    doc = fitz.open(SRC_DIR / "ipc.pdf")
    pages_text = [p.get_text() for p in doc]
    
    # Body starts around page 15 (after Table of Contents)
    body_text = "\n".join(pages_text[13:])
    
    # Clean non-breaking and weird quotes/characters
    body_text = body_text.replace("\x00", "").replace("\r", "")
    
    # Regex to match sections in body: e.g. "\n302. Punishment for murder.—Whoever..."
    pattern = re.compile(
        r'(?:\n|\A)\s*(\d+[A-Z]?)\.\s+([^\n—\.-]+)[—\.-]\s*(.+?)(?=(?:\n\s*\d+[A-Z]?\.\s+[^\n—\.-]+[—\.-])|\Z)',
        re.DOTALL
    )
    
    ipc_dict = {}
    for match in pattern.finditer(body_text):
        sec_num = match.group(1).strip()
        title = match.group(2).strip()
        body = match.group(3).strip()
        
        # Clean text
        clean_text = f"{title}.—{body}"
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) > 20 and sec_num not in ipc_dict:
            ipc_dict[sec_num] = {
                "law": "IPC",
                "section_id": sec_num,
                "title": title,
                "section_text": clean_text,
                "offence_type": classify_offence_type(clean_text, title),
                "source_file": "ipc.pdf"
            }
            
    print(f"Extracted {len(ipc_dict)} IPC sections from PDF")
    return ipc_dict


def extract_bns_sections() -> dict[str, dict]:
    bns_extracted_path = OUTPUT_DIR / "bns_sections_extracted.json"
    if not bns_extracted_path.exists():
        raise FileNotFoundError(f"{bns_extracted_path} not found")
        
    with open(bns_extracted_path, "r", encoding="utf-8") as f:
        bns_data = json.load(f)
        
    bns_dict = {}
    for sec_num, sec_text in bns_data.items():
        clean_text = re.sub(r'\s+', ' ', sec_text).strip()
        bns_dict[sec_num] = {
            "law": "BNS",
            "section_id": sec_num,
            "title": f"BNS Section {sec_num}",
            "section_text": clean_text,
            "offence_type": classify_offence_type(clean_text),
            "source_file": "bns.pdf"
        }
    print(f"Loaded {len(bns_dict)} BNS sections from extracted JSON")
    return bns_dict


def main():
    print("=" * 80)
    print("BUILDING STATUTE DATASET & PINECONE VECTORS")
    print("=" * 80)
    
    ipc_sections = extract_ipc_sections()
    bns_sections = extract_bns_sections()
    
    chunks_path = OUTPUT_DIR / "statute_chunks_complete.jsonl"
    all_chunks = []
    
    with open(chunks_path, "w", encoding="utf-8") as f:
        for sec_num, data in ipc_sections.items():
            chunk = {
                "chunk_id": f"IPC_{sec_num}",
                "law": "IPC",
                "section_id": sec_num,
                "section_text": data["section_text"],
                "full_text": data["section_text"],
                "offence_type": data["offence_type"],
                "source_file": "ipc.pdf"
            }
            all_chunks.append(chunk)
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
        for sec_num, data in bns_sections.items():
            chunk = {
                "chunk_id": f"BNS_{sec_num}",
                "law": "BNS",
                "section_id": sec_num,
                "section_text": data["section_text"],
                "full_text": data["section_text"],
                "offence_type": data["offence_type"],
                "source_file": "bns.pdf"
            }
            all_chunks.append(chunk)
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    print(f"✓ Saved {len(all_chunks)} total statute chunks to {chunks_path}")
    
    # Generate vectors
    print("\nLoading SentenceTransformer('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    vectors_path = OUTPUT_DIR / "statute_vectors.jsonl"
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    
    texts = [c["section_text"][:500] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    
    with open(vectors_path, "w", encoding="utf-8") as f:
        for chunk, emb in zip(all_chunks, embeddings):
            record = {
                "chunk_id": chunk["chunk_id"],
                "law": chunk["law"],
                "section_id": chunk["section_id"],
                "offence_type": chunk["offence_type"],
                "source_file": chunk["source_file"],
                "embedding": emb.tolist()
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"✓ Saved {len(all_chunks)} vectors to {vectors_path}")
    
    # Deploy to Pinecone
    print("\nDeploying to Pinecone...")
    from deploy_to_pinecone import deploy_to_pinecone
    deploy_to_pinecone()


if __name__ == "__main__":
    main()
