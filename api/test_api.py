import sys
import requests
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_URL = "http://localhost:8080/api/v1/predict"  # Default API endpoint (falls back to port 8000)
# ==============================================================================

# Diverse built-in test cases for quick testing
EXAMPLE_TEXTS = [
    {
        "type": "Depression (Severe)",
        "text": "I feel so hollow and empty. I do not see any reason to keep trying. Everything is completely hopeless and I just want the pain to end."
    },
    {
        "type": "Depression (Mild/Borderline)",
        "text": "Lately I have just been feeling a bit down and unmotivated. I don't feel like doing my usual hobbies, but I'm still getting through my day."
    },
    {
        "type": "Control (Healthy/Normal)",
        "text": "Had a wonderful day at the park today! The weather was perfect and the dogs loved running around. Feeling very grateful."
    },
    {
        "type": "Control (Stress but Normal)",
        "text": "Studying for my exams next week. Feeling a bit stressed and tired, but I think I am prepared. Just going to keep working hard."
    }
]

def print_box(prob, pred, label, threshold, text):
    color_code = "\033[91m" if pred == 1 else "\033[92m"  # Red for Depression, Green for Control
    reset_code = "\033[0m"
    bold_code = "\033[1m"
    
    print("\n" + "=" * 80)
    print(f" {bold_code}PREDICTION RESULT{reset_code} ".center(88, "="))
    print("=" * 80)
    print(f"Input Text : {text}")
    print("-" * 80)
    print(f"Probability: {prob:.4f}")
    print(f"Threshold  : {threshold:.3f}")
    print(f"Prediction : {pred}")
    print(f"Label      : {color_code}{bold_code}{label}{reset_code}")
    print("=" * 80 + "\n")

def query_api(text, url=API_URL):
    try:
        response = requests.post(url, json={"text": text}, timeout=5)
        if response.status_code == 200:
            res_data = response.json()
            print_box(
                prob=res_data["depression_probability"],
                pred=res_data["depression_prediction"],
                label=res_data["label"],
                threshold=res_data["threshold_used"],
                text=res_data["text"]
            )
        else:
            print(f"\033[91mError: API returned status code {response.status_code}\033[0m")
            print(response.text)
    except requests.exceptions.ConnectionError:
        # Fallback to port 8000 if port 8080 fails
        if "8080" in url:
            fallback_url = url.replace("8080", "8000")
            print(f"Port 8080 connection failed. Trying fallback endpoint: {fallback_url}...")
            query_api(text, fallback_url)
        else:
            print(f"\033[91mError: Could not connect to API at {url}.\033[0m")
            print("Please ensure your API server is running (e.g. uvicorn api.main:app).")

def main():
    print("=========================================================")
    print("      Interactive Depression API Tester Client           ")
    print("=========================================================")
    
    while True:
        print("\nSelect an option:")
        print("1. Enter custom text to predict")
        print("2. Test with built-in example texts")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            text = input("\nEnter text to validate: ").strip()
            if text:
                query_api(text)
            else:
                print("Text cannot be empty.")
                
        elif choice == "2":
            print("\nSelect an example text:")
            for idx, item in enumerate(EXAMPLE_TEXTS):
                print(f"{idx + 1}. [{item['type']}] {item['text'][:65]}...")
            
            ex_choice = input(f"Enter choice (1-{len(EXAMPLE_TEXTS)}): ").strip()
            try:
                ex_idx = int(ex_choice) - 1
                if 0 <= ex_idx < len(EXAMPLE_TEXTS):
                    selected = EXAMPLE_TEXTS[ex_idx]
                    query_api(selected["text"])
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid input.")
                
        elif choice == "3":
            print("\nExiting. Thank you!")
            break
        else:
            print("Invalid option. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
