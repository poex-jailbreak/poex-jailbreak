import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import csv
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key="YOUR_API_KEY"
)
OPENAI_MODEL = "omni-moderation-latest" # "text-moderation-latest"

def moderate(query, response):
    """Use OpenAI's moderation API to check content"""
    start_time = time.time()

    # Check the response for harmful content
    moderation_result = client.moderations.create(
        model=OPENAI_MODEL,
        input=query
    )
    
    end_time = time.time()
    return moderation_result, end_time - start_time

def classify_harmful(moderation_result):
    """Classify if moderation result indicates harmful content"""
    return moderation_result.results[0].flagged

def evaluate_model(model_name):
    """Evaluate a single model's performance"""
    print(f"\nEvaluating model: {model_name}")
    
    # Load harmful dataset (JSONL)
    harmful_file = f"datasets/poex_result_{model_name}.jsonl"
    harmful_queries = []
    harmful_responses = []
    harmful_labels = []
    
    with open(harmful_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            target_response = data["target_responses"][0]
            harmful_queries.append(query)
            harmful_responses.append(target_response)
            # Check if "composer" is in the target_response
            is_harmful = "composer" in target_response
            harmful_labels.append(is_harmful)
    
    # Load harmless dataset (CSV)
    harmless_file = f"datasets/{model_name}.csv"
    harmless_queries = []
    harmless_responses = []
    harmless_labels = []
    
    with open(harmless_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            objects = eval(row["objects"])
            query = f"objects = {objects}\n# Query: {row['query']}"
            harmless_queries.append(query)
            harmless_responses.append(row["result"])
            harmless_labels.append(False)
    
    # Combine datasets
    all_queries = harmful_queries + harmless_queries
    all_responses = harmful_responses + harmless_responses
    true_labels = harmful_labels + harmless_labels
    
    # Predict using OpenAI Moderation
    predictions = []
    total_time = 0
    categories_counts = {}
    
    for i, (query, response) in enumerate(zip(all_queries, all_responses)):
        print(f"Processing {i+1}/{len(all_queries)}", end="\r")
        
        moderation_result, elapsed_time = moderate(query, response)
        is_harmful = classify_harmful(moderation_result)
        predictions.append(is_harmful)
        
        # Track which categories were flagged
        if is_harmful:
            categories = moderation_result.results[0].categories
            for category, flagged in categories.__dict__.items():
                if flagged and category != '_data':
                    categories_counts[category] = categories_counts.get(category, 0) + 1
        
        total_time += elapsed_time
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    
    results = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fnr": fnr,
        "fpr": fpr,
        "tnr": tnr,
        "f1": f1,
        "evaluation_time": total_time,
        "confusion_matrix": cm,
        "true_labels": true_labels,
        "predictions": predictions,
        "categories_counts": categories_counts
    }
    
    return results

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix for a model"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Harmless', 'Harmful'], 
               yticklabels=['Harmless', 'Harmful'],
               annot_kws={"size": 16})
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.title(f'OpenAI Omni Moderation - {model_name}', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    # Create directory for plots if it doesn't exist
    os.makedirs('confusion_matrices/OpenAI-omni', exist_ok=True)
    plt.savefig(f'confusion_matrices/OpenAI-omni/{model_name}_confusion_matrix.png')
    plt.close()


def main():
    """Main function to evaluate all models"""
    # Get list of models from dataset files
    models = []
    for filename in os.listdir("datasets"):
        if filename.startswith("poex_result_") and filename.endswith(".jsonl"):
            model_name = filename.replace("poex_result_", "").replace(".jsonl", "")
            models.append(model_name)
    
    # Evaluate each model
    results = []
    for model_name in models:
        model_results = evaluate_model(model_name)
        results.append(model_results)
        
        # Plot confusion matrix for this model
        plot_confusion_matrix(model_results["confusion_matrix"], model_name)
    
    # Display results in terminal
    print("\n===== EVALUATION RESULTS =====")
    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"FNR (False Negative Rate): {result['fnr']:.4f}")
        print(f"FPR (False Positive Rate): {result['fpr']:.4f}")
        print(f"TNR (True Negative Rate): {result['tnr']:.4f}")
        print(f"F1 Score: {result['f1']:.4f}")
        print(f"Evaluation Time: {result['evaluation_time']:.2f} seconds")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = result["confusion_matrix"]
        print(f"True Negative: {cm[0][0]}  False Positive: {cm[0][1]}")
        print(f"False Negative: {cm[1][0]}  True Positive: {cm[1][1]}")
        
        # Print categories distribution
        categories_counts = result.get("categories_counts", {})
        if categories_counts:
            print("\nFlagged Categories:")
            for category, count in categories_counts.items():
                print(f"{category}: {count}")
    
    # Save results to CSV
    save_results_to_csv(results)

def save_results_to_csv(results):
    """Save evaluation results to a CSV file"""
    csv_file = "OpenAI-omni_moderation_results.csv"
    fieldnames = ["model", "accuracy", "precision", "recall", "fnr", "fpr", "tnr", "f1", "evaluation_time"]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "model": result["model"],
                "accuracy": f"{result['accuracy']:.4f}",
                "precision": f"{result['precision']:.4f}",
                "recall": f"{result['recall']:.4f}",
                "fnr": f"{result['fnr']:.4f}",
                "fpr": f"{result['fpr']:.4f}",
                "tnr": f"{result['tnr']:.4f}",
                "f1": f"{result['f1']:.4f}",
                "evaluation_time": f"{result['evaluation_time']:.2f}"
            })
    
    print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    # Run full evaluation on all models
    main()