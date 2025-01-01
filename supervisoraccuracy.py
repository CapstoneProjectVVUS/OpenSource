import pandas as pd

# Load the CSV file
# hermes38b-llama3.1-fp16\questions_and_responses_hermes3_8b-llama3.1-fp16.csv
file_path = 'questions_and_responses_qwen2.5_32b-instruct-q6_K.csv'
df = pd.read_csv(file_path)

# Define the correct agent mapping based on question index ranges
correct_agent_mapping = {
    range(0, 20): "Tennis",
    range(20, 40): "Skateboarding",
    range(40, 60): "Schedule",
    range(60, 80): "Other",
    range(80, 100): "Archery"
}

# Add a column for the correct agent based on question index
def determine_correct_agent(index):
    for key_range, agent in correct_agent_mapping.items():
        if index in key_range:
            return agent
    return None

df['correct_agent'] = df.index.map(determine_correct_agent)

# Calculate the accuracy of the supervisor in calling the right agent
df['is_correct'] = df['Agent'] == df['correct_agent']
accuracy_score = df['is_correct'].sum()

print(f"Accuracy Score: {accuracy_score}/100")
