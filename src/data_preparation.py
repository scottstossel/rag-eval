import json
import numpy as np
from collections import Counter
from datasets import load_dataset

dataset = load_dataset("squad", split="train[:1000]")
print("Dataset length: ", len(dataset))

unique_contexts = list(set(dataset["context"]))
print("Unique contexts: ", len(unique_contexts))
print("Unique questions: ", len(set(dataset["question"])))

context_lengths = [len(context.split()) for context in dataset["context"]]
print("Mean context length: ", np.mean(context_lengths))
print("Max context length: ", np.max(context_lengths))
print("Min context length: ", np.min(context_lengths))

question_lengths = [len(question.split()) for question in dataset["question"]]
print("Mean question length: ", np.mean(question_lengths))

context_word_counts = Counter(dataset["context"])
counts = list(context_word_counts.values())
print("Min Q per context:", min(counts))
print("Max Q per context:", max(counts))
print("Mean Q per context:", np.mean(counts))

def check_answer_in_context(row):
    answer = row["answers"]["text"][0]
    return answer in row["context"]

presence = [check_answer_in_context(r) for r in dataset]
print("Answer present rate: ", sum(presence) / len(presence))

# Build documents
documents = []
context_to_doc_id = {}
for i, context in enumerate(unique_contexts):
    doc_id = f"doc_{i}"
    documents.append({"doc_id": doc_id, "text": context})
    context_to_doc_id[context] = doc_id

# Build QA pairs
qa_pairs = []
for row in dataset:
    qa_pairs.append({
        "question": row["question"],
        "answer": row["answers"]["text"][0],
        "gold_doc_id": context_to_doc_id[row["context"]]
    })

# Save outputs
with open("outputs/documents.json", "w") as f:
    json.dump(documents, f, indent=2)

with open("outputs/qa_pairs.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

with open("outputs/context_to_doc_id.json", "w") as f:
    json.dump(context_to_doc_id, f, indent=2)

print("Saved documents, qa_pairs, and context_to_doc_id to outputs/")