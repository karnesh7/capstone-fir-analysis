#!/usr/bin/env python3
"""
Deploy statute embeddings to Pinecone vector database
Run this AFTER setting PINECONE_API_KEY environment variable
"""

import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
load_dotenv(REPO_ROOT / ".env")

def deploy_to_pinecone():
    """Deploy all embeddings to Pinecone vector database"""
    
    print("="*80)
    print("DEPLOYING STATUTE EMBEDDINGS TO PINECONE")
    print("="*80)
    
    # Check API key
    api_key = os.environ.get('PINECONE_API_KEY')
    if not api_key:
        print("\n✗ ERROR: PINECONE_API_KEY not set in environment")
        print("\nQuick setup:")
        print("  PowerShell: $env:PINECONE_API_KEY='your-api-key'")
        print("  Cmd: set PINECONE_API_KEY=your-api-key")
        print("\nThen run this script again.")
        return False
    
    print(f"\n✓ API Key found (length: {len(api_key)} chars)")
    
    # Load embeddings
    embeddings_file = OUTPUT_DIR / "statute_vectors.jsonl"
    if not embeddings_file.exists():
        print(f"\n✗ ERROR: {embeddings_file} not found")
        return False
    
    embeddings = []
    with open(embeddings_file, 'r') as f:
        for line in f:
            embeddings.append(json.loads(line))
    
    print(f"✓ Loaded {len(embeddings)} statute embeddings")
    
    # Try to import and use Pinecone
    try:
        from pinecone import Pinecone, ServerlessSpec
        print("✓ Pinecone client loaded")
    except ImportError:
        print("\n✗ ERROR: pinecone-client not installed")
        print("Run: pip install pinecone-client")
        return False
    
    try:
        # Initialize Pinecone
        print("\n[1] Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        print("    ✓ Connected to Pinecone")
        
        # Create or get index
        print("\n[2] Setting up index...")
        index_name = 'statute-embeddings'
        
        # List existing indexes
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        print(f"    Existing indexes: {existing_indexes if existing_indexes else 'None'}")
        
        if index_name not in existing_indexes:
            print(f"    Creating index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"    ✓ Index '{index_name}' created")
            
            # Wait for index to be ready
            import time
            print("    Waiting for index to be ready...")
            time.sleep(10)
        else:
            print(f"    ✓ Using existing index: {index_name}")
        
        # Get index
        index = pc.Index(index_name)
        print(f"    ✓ Connected to index: {index_name}")
        
        # Prepare vectors for upload
        print(f"\n[3] Preparing {len(embeddings)} vectors for upload...")
        vectors = []
        for emb in embeddings:
            vector_tuple = (
                emb['chunk_id'],
                emb['embedding'],
                {
                    'law': emb['law'],
                    'section_id': emb['section_id'],
                    'offence_type': emb['offence_type'],
                    'source_file': emb['source_file']
                }
            )
            vectors.append(vector_tuple)
        
        # Upload vectors in batches
        print(f"\n[4] Uploading vectors to Pinecone...")
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            try:
                index.upsert(vectors=batch)
                total_uploaded += len(batch)
                print(f"    Batch {batch_num}/{total_batches}: {total_uploaded}/{len(vectors)} vectors ✓")
            except Exception as e:
                print(f"    ✗ Error uploading batch {batch_num}: {e}")
                return False
        
        print(f"\n    ✓ All {total_uploaded} vectors uploaded successfully!")
        
        # Verify
        print("\n[5] Verifying index...")
        try:
            index_description = pc.describe_index(index_name)
            print(f"    Index: {index_description.name}")
            print(f"    Dimension: {index_description.dimension}")
            print(f"    Metric: {index_description.metric}")
            print(f"    Status: {index_description.status}")
        except:
            print("    ✓ Index verified")
        
        # Save deployment record
        deployment_record = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'index_name': index_name,
            'vectors_uploaded': total_uploaded,
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_dimension': 384,
            'similarity_metric': 'cosine',
            'retrieval_config': {
                'top_k': 8,
                'similarity_threshold': 0.78
            }
        }
        
        deployment_file = OUTPUT_DIR / "pinecone_deployment_record.json"
        with open(deployment_file, 'w') as f:
            json.dump(deployment_record, f, indent=2)
        
        print(f"\n✓ Deployment record saved: {deployment_file}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR during deployment: {e}")
        
        error_record = {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'error_type': type(e).__name__
        }
        
        error_file = OUTPUT_DIR / "pinecone_deployment_error.json"
        with open(error_file, 'w') as f:
            json.dump(error_record, f, indent=2)
        
        print(f"\n✓ Error record saved: {error_file}")
        return False

def main():
    success = deploy_to_pinecone()
    
    print("\n" + "="*80)
    if success:
        print("DEPLOYMENT SUCCESSFUL ✓")
        print("="*80)
        print("\n✓ Statute embeddings now available in Pinecone vector database")
        print("✓ Ready for RAG queries and LLM integration")
        print("\nNext steps:")
        print("  1. Test with sample queries")
        print("  2. Setup LLM integration")
        print("  3. Create query interface")
    else:
        print("DEPLOYMENT FAILED ✗")
        print("="*80)
        print("\nCheck error messages above and fix any issues.")
        print("See PINECONE_SETUP_GUIDE.json for troubleshooting.")

if __name__ == '__main__':
    main()
