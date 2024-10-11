import pandas as pd
from transformers import pipeline

# Load the dataset 
dataset = pd.read_csv('improved_synthetic_email_dataset.csv')

 
generator = pipeline('text-generation', model='gpt2',device=0)


def generate_reply(email):
    try:
       
        response = generator(email, max_new_tokens=100, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.9)[0]['generated_text']
        return response
    except Exception as e:
        print(f"Error generating reply for email: {email}")
        print(f"Error: {e}")
        return "Could not generate a response."


dataset['Reply'] = dataset['Email'].apply(generate_reply)

# Save the dataset with the generated replies
dataset.to_csv('synthetic_email_replies.csv', index=False)

print("Replies generated and saved to 'synthetic_email_replies.csv'.")
