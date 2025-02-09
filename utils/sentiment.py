import ollama

def analyze_sentiment(text):
    model = "llama3.2:latest"  # Change this to your preferred Ollama model
    prompt = f"Analyze the sentiment of the following text and provide a list of human moods assigned and a score for each on a scale of 0 to 1.\n\nText: {text}"
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"]

def analyze_emotion(text):
    model = "llama3.2:latest"  # Change this to your preferred Ollama model
    prompt = (f"Based on the given analysis and scores based on 0 to 1 for each emotion give me the mood (Human emotion) of the person and also an emoji for it.\n\nText: {text}" 
                "Only give me the mood in one word and the emoji based on these labels: Anger, Disgust, Fear, Happiness, Sadness, Surprise, Contempt, Pride, Shame, Embarrassment, Excitement, Joy, Trust, Anticipation, Love, Guilt, Awe, Interest, Curiosity"
                "Do not mention the label number but only one of the vales for each label"
                "The response MUST be one word and one emoji"
    )
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"]

# if __name__ == "__main__":
#     user_text = "Dragged myself out of bed, grabbed a disgustingly overpriced coffee, and tried to read a paper on SVM optimizations. I swear, some of these authors just write words to flex their vocabulary. Could barely understand half of it."
#     sentiment_result = analyze_sentiment(user_text)
#     print("Sentiment Analysis Result:")
#     print(sentiment_result)
#     print("########END######")
#     mood_result = analyze_emotion(sentiment_result)
#     print(mood_result)

