# -------------------------------------------------------
# AI Quiz Master using Naive Bayes
# -------------------------------------------------------
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

# ---------- Build classifier ----------
def build_classifier(positives, negatives):
    texts = positives + negatives
    labels = [1]*len(positives) + [0]*len(negatives)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(texts, labels)
    return model

# ---------- Main Quiz ----------
def main():
    print("WELCOME TO AI QUIZ MASTER")
    name = input("Enter your name: ").strip().title()
    print(f"\nHello, {name}! Let's begin your AI quiz.\n")

    quiz = [
        {
            "q": "What does AI stand for?",
            "a": "Artificial Intelligence",
            "positives": ["artificial intelligence", "ai means artificial intelligence"],
            "negatives": ["machine learning", "artificial information"]
        },
        {
            "q": "Name one popular library used in Python for AI.",
            "a": "TensorFlow",
            "positives": ["tensorflow", "keras", "pytorch"],
            "negatives": ["numpy", "pandas", "matplotlib"]
        },
        {
            "q": "What is Machine Learning in AI?",
            "a": "Training machines using data and experience",
            "positives": [
                "training machines using data",
                "training models with experience",
                "learn from data"
            ],
            "negatives": [
                "ai that plays chess",
                "collecting random data",
                "storing programs"
            ]
        },
        {
            "q": "Which field of AI deals with understanding human language?",
            "a": "Natural Language Processing",
            "positives": ["natural language processing", "nlp", "human language understanding"],
            "negatives": ["computer vision", "robotics", "speech synthesis"]
        },
        {
            "q": "What is the full form of NLP in AI?",
            "a": "Natural Language Processing",
            "positives": ["natural language processing", "nlp full form"],
            "negatives": ["neural learning process", "network layer protocol"]
        }
    ]

    score = 0
    results = []

    for i, q in enumerate(quiz, start=1):
        clf = build_classifier(q["positives"], q["negatives"])
        print(f"Q{i}: {q['q']}")
        ans = input("Your answer: ").strip().lower()
        pred = clf.predict([ans])[0]

        if pred == 1:
            print("Correct!\n")
            score += 1
            results.append(f"Q{i}: Correct ({ans})")
        else:
            print(f"Wrong! Correct answer: {q['a']}\n")
            results.append(f"Q{i}: Wrong ({ans}) -> Correct: {q['a']}")

    # ---------- Summary ----------
    accuracy = (score / len(quiz)) * 100
    print("-----------------------------------------")
    print(f"Quiz Completed! Final Score: {score}/{len(quiz)}")
    print(f"Accuracy: {accuracy:.2f}%")

    if accuracy == 100:
        print("Excellent! Perfect score!")
    elif accuracy >= 80:
        print("Good job! Keep learning AI concepts.")
    else:
        print("Nice try! Review your AI basics.")

    # ---------- Save results ----------
    with open("quiz_results.txt", "w", encoding="utf-8") as f:
        f.write(f"AI QUIZ RESULTS - {name}\n")
        f.write("\n".join(results))
        f.write(f"\n\nFinal Score: {score}/{len(quiz)} ({accuracy:.2f}%)")

    print("\nYour score has been saved to 'quiz_results.txt'.")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()

