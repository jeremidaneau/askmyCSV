import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Page title
st.set_page_config(page_title='Ask the 📄 ')
st.title('Ask the 📄 ')

# Load CSV file
def load_csv(input_csv):
  df = pd.read_csv(input_csv)
  with st.expander('See DataFrame'):
    st.write(df)
  return df

# Calculate token count function
def calculate_token_count(text):
    # Split the text into tokens (you might need to adjust this depending on your tokenizer)
    tokens = text.split()
    return len(tokens)

# Generate LLM response
def generate_response(csv_file, input_query, max_tokens):
    llm = ChatOpenAI(model_name='gpt-4-32k', temperature=0.2, openai_api_key=openai_api_key)
    df = load_csv(csv_file)
    
    # Calculate token count of the input query
    input_query_tokens = calculate_token_count(input_query)
    
    # Calculate max_tokens value by subtracting input query tokens from the desired maximum token count
    max_tokens -= input_query_tokens
    
    # Create Pandas DataFrame Agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
    
    # Perform Query using the Agent with max_tokens
    response = agent.run(input_query)
    
    return st.success(response)

# Input widgets
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
question_list = [
  'How many rows are there?',
  'What is the range of values for MolWt with logS greater than 0?',
  'How many rows have MolLogP value greater than 0.',
  'Other']
query_text = st.selectbox('Select an example query:', question_list, disabled=not uploaded_file)
openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))

# App logic
if query_text == 'Other':
  query_text = st.text_input('Enter your query:', placeholder='Enter query here ...', disabled=not uploaded_file)
if not openai_api_key.startswith('sk-'):
  st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-') and (uploaded_file is not None):
  st.header('Output')
  
  # Specify the desired maximum token count here (e.g., 50 tokens)
  max_tokens = 50
  
  generate_response(uploaded_file, query_text, max_tokens)
