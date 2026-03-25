from google.cloud import storage
import os

client = storage.Client()
bucket = client.get_bucket("mhw-risk-cache")
blobs  = list(client.list_blobs("mhw-risk-cache", max_results=10))
print(f"✅ Auth OK  | Project: {client.project}")
print(f"✅ Bucket   | {bucket.name} @ {bucket.location} ({bucket.storage_class})")
print(f"✅ Contents | {len(blobs)} object(s) found" if blobs else "✅ Contents | Bucket is empty (expected for new bucket)")
