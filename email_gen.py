import random
import pandas as pd
from transformers import pipeline


generator = pipeline('text-generation', model='gpt2', device=0)  

# Define email categories
categories = ['Student', 'Corporate', 'Researcher']

# Updated prompts for each category
prompts = {
    'Student': [
        "Dear HOD, I have some concerns regarding my academic performance this semester, and I would appreciate your guidance on how to improve.",
        "Hello Professor, I would like to inquire about the process for applying for an internship. Can you provide me with the necessary details?",
        "Sir/Madam, I missed the last lecture due to health reasons. Could you kindly share the course material or any important updates?",
        "Dear Professor, I am working on my final year project and need some clarification regarding the submission guidelines.",
        "Respected Sir/Madam, I am unsure about the elective course selection process for the next semester. Can you guide me through it?",
        "Hello Professor, I need your assistance in understanding the grading criteria for the mid-semester exams.",
        "Dear HOD, could you please help me with a recommendation letter for a scholarship application?",
        "Sir/Madam, I would like to request an extension for my assignment due to personal reasons. Can this be accommodated?"
    ],
    'Corporate': [
        "Dear HOD, we are interested in exploring partnership opportunities for recruiting students from your department. Could we schedule a meeting to discuss this further?",
        "Hello Professor, we would like to organize a placement drive at your university. Could you provide information on the process and available dates?",
        "Dear Sir/Madam, our company is looking for interns with a strong background in mechanical engineering. Could you recommend any students for the upcoming internship season?",
        "Dear Professor, we would like to collaborate on a research project related to AI and robotics. Can we discuss potential areas for cooperation?",
        "Hello, we are organizing a tech summit next month and would like to invite you and your students to participate. Could you share the details with your department?",
        "Dear HOD, we are conducting a campus recruitment program and would like to know more about the student profiles and their availability for placements.",
        "Dear Professor, our firm is looking for opportunities to sponsor research projects in your field. Could we have a conversation about potential funding?"
    ],
    'Researcher': [
        "Dear HOD, I am working on a research project that overlaps with your department’s focus on material science. Would you be open to a collaboration?",
        "Hello Professor, I came across your recent publication on quantum computing, and I would like to discuss the possibility of using your facilities for extended research.",
        "Dear Professor, I would like to request access to your department’s shared dataset on neural networks for a comparative study. Can this be arranged?",
        "Dear HOD, I am interested in applying for a research position in your lab. Could you share information on the current openings and requirements?",
        "Dear Sir/Madam, I would appreciate any insights you could provide on the methodology used in your recent research on renewable energy solutions.",
        "Hello Professor, I am organizing a workshop on computational biology and would like to invite you and your team to present your research. Can you confirm your availability?",
        "Dear Professor, I would like to inquire about the possibility of utilizing your department’s lab facilities for a joint research project on nanotechnology.",
        "Hello, I am preparing a research paper on deep learning algorithms and would love to get your input or suggestions based on your department's expertise."
    ]
}

# Generate synthetic emails with improved generation settings
def generate_email(category, num_emails=5):
    prompt = random.choice(prompts[category])
    generated_emails = generator(
        prompt,
        max_length=150,  
        num_return_sequences=num_emails,
        temperature=0.7, 
        top_k=50,  
        top_p=0.9, 
        repetition_penalty=1.2,  
        truncation=True  
    )
    for emails in generated_emails:
      print(emails)

    return [email['generated_text'] for email in generated_emails]

# Create a dataset
def create_dataset(num_samples_per_category=100):
    data = []
    for category in categories:
        for _ in range(num_samples_per_category):
            email = generate_email(category, num_emails=1)[0]
            data.append({'Category': category, 'Email': email})
    return pd.DataFrame(data)

# Generate the dataset
email_dataset = create_dataset()

# Save the dataset to CSV
email_dataset.to_csv('improved_synthetic_email_dataset.csv', index=False)
print("Improved synthetic email dataset generated and saved to 'improved_synthetic_email_dataset.csv'.")
