# =========================================================
# Created by Nguyen Quoc Thai
# Date: 08/05/2024
# This file contains functions for a chatbot application using the Llama 2 model. It includes functions for extracting information from conversations, saving chat history with sentiment analysis results, clearing chat history, and generating responses using the Llama 2 model.
# Functions:
#   extract_information(conversation, pattern): Extracts factors from a conversation based on a given regex pattern.
#   save_chat_sentiment_history(): Saves the chat history and the sentiment result for each pair of chatbot-user conversation.
#   clear_chat_history(): Cleans the chat history.
#   generate_llama2_response(prompt_input, system_promt, extract=False): Generates a response using the Llama 2 model.
#   main(): perform the Streamlit and chatbot.
# =========================================================
import re
import os
import csv
import replicate
import streamlit as st
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_information(conversation, pattern):
    """Extract the factor from conversation

    Args:
        conversation (list): list contains assistant chat and user chat
        pattern (regex equation string): regex format to find the pattern.

    Returns:
        string/None: the data extraction if found or None.
    """

    match = re.search(pattern, conversation, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

def save_chat_sentiment_history():
    """Save the chat history and the sentiment result for each pair of chatbot-user conversation.
    """
    st.session_state.messages

    assistant = [item['content'] for item in st.session_state.messages if item['role'] == 'assistant']
    user = [item['content'] for item in st.session_state.messages if item['role'] == 'user']
    label = [item['content'] for item in st.session_state.sentiment_score if item['role'] == 'assistant']
    score = [item['content'] for item in st.session_state.sentiment_score if item['role'] == 'user']

    with open(f'./data/chat_history_{st.session_state.chat_index}.csv', mode='w', newline='') as file:
        fieldnames = ['assistant', 'user', 'label', 'score']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write data
        for i in range(len(assistant)):
            writer.writerow({'assistant': assistant[i],
                            'user': user[i],
                            'label': label[i],
                            'score': score[i]})
    st.session_state.chat_index+=1

def clear_chat_history():
    """Clean chat history function
    """
    save_chat_sentiment_history()
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.sentiment_score = []
    st.session_state.flag_enough = 0
    st.session_state.thresh = 7
    st.session_state.infor ={
        "City": None,
        "Budget": None,
        "Bedroom": None,
        "Bathroom": None,
        "Floors": None,
        "Facilities" : None,
        "Noise_level": None,
        "Crime_rate" : None,
        "Properties": None,
        "Addtional_information":None

    }

def generate_llama2_response(prompt_input, system_promt, extract=False):
    """Generate the llama2 respone

    Args:
        prompt_input (string): the prompt input from user.
        system_promt (string): system promt for the model.
        extract (bool): extract the information with predefine dict. Default is F

    Returns:
        output (list): a repsonse from the llama model chatbot.
    """
  
    # string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. Only reply with role 'Assistant'."
    
    
    # Extract information from user input
    if extract == True:
        string_dialogue = ''
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
            
        for item in st.session_state.infor:
            if st.session_state.infor[item] == None:
                string_dialogue += f"Find {item} mention this text"
                mini_system_promt = "You are an extractor information tool, do not generate user responses.\
                Only return as:\
                {item}: None\
                If you do not found, or:\
                {item}: {item} name you found"

                extract_output = replicate.run(llm, 
                                input={"top_p":top_p,
                                        "prompt": f"{string_dialogue}",
                                        "temperature":temperature,
                                        "system_prompt": mini_system_promt,
                                        "max_new_tokens": 50,
                                        "repetition_penalty": 1
                                        })
                full_response = ''
                for item in extract_output:
                    full_response += item
                
                pattern = rf'\b{item}: \s*(.*)'
                extract_data = extract_information(full_response, pattern)
                if extract_data != None:
                    st.session_state.infor[item] = str(extract_data)
                    st.session_state.flag_enough += 1

    output = replicate.run(llm, 
                           input={"top_p":top_p,
                                  "prompt": f"{prompt_input}",
                                  "temperature":temperature,
                                  "system_prompt": system_promt,
                                  "max_new_tokens": max_length,
                                  "repetition_penalty": 1
                                })
    return output


if __name__ == '__main__':

    # Index to store the chat later
    st.session_state.chat_index = 0

    # List to save the sentiment analysis of each pair assistant - user chat
    st.session_state.sentiment_res = []

    # Feature extraction count
    st.session_state.flag_enough = 0
    st.session_state.flag_check = 0

    # Threshold that evaluate as enough data extraction
    st.session_state.thresh = 7

    # Store the information extract from user
    st.session_state.infor = {
        "City": None,
        "Budget": None,
        "Bedroom": None,
        "Bathroom": None,
        "Floors": None,
        "Facilities" : None,
        "Noise_level": None,
        "Crime_rate" : None,
        "Properties": None,
        "Addtional_information":None
    }

    # App title
    st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

    # Please replace by your Replicate api token since mine is expired :((((
    os.environ['REPLICATE_API_TOKEN'] = "r8_***********************************"
    replicate_api = "r8_***********************************"

    # Replicate Credentials
    with st.sidebar:
        st.title('🦙💬 Llama 2 Chatbot')

        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama3-8B'], key='selected_model')
        if selected_model == 'Llama3-8B':
            # llm = 'meta/meta-llama-3-8b'
            llm = 'meta/llama-2-7b-chat'
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.7, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.number_input("Insert max length for the response...", min_value = 100)

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", 
        "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Clean chat history
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Create a sentiment analysis from hugging face
    sentiment_pipeline = pipeline("sentiment-analysis")

    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.sentiment_res.extend(sentiment_pipeline(prompt))
    
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant" \
        and st.session_state.flag_enough < st.session_state.thresh:
        system_promt = """Do not generate user responses on your own and avoid repeating questions. You are an in-depth property analysis and suggestion assistant. You try to help people find an ideal available house. Whenever chat with user if they want you to find a house or place to live, you will try to ask them some factor but do not mention them directly: city, budget, number of bedrooms and bathroom, Facilities nearby(example: park, school, hospital, supper market,..), Noise level, Crime rate, Property belongs, Addtional information...
Collect all of the information one by one and re-confirm each information you get as the format:
          
    Ctiy:
    Budget
    Bedroom:
    Bathroom:
    Facilities: 
    Noise_level 
    Crime_rate: 
    Properties:
    Addtional information:

    And do not ask again if user has stated it in the conversation before. Respond with just 'Thank you for choosing us' at the end.  
    """
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt, system_promt, extract=True)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

    # Retrive from dataset
    if st.session_state.flag_enough >= st.session_state.thresh and st.session_state.flag_check == 0:
        st.session_state.flag_check = 1
        st.session_state.query_data = ''
        for item in st.session_state.infor:
            if st.session_state.infor[item] != None:
                st.session_state.query_data += f' {st.session_state.infor[item]}'

        # Load the faiss data
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
        db_load = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        # Retrieve top 4 place nearly same context with user input
        retrieval_data_db = db_load.similarity_search(st.session_state.query_data, k=4)
        
        st.session_state.retrieval_data = ''
        for i in retrieval_data_db:
            st.session_state.retrieval_data += f'{i.page_content}\n'
        
        # Return list of suggestion    
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(
                    f"For each place and it information try to advertise it",
                    f"You are an advertised assistant, base on the 4 property information : {st.session_state.retrieval_data}, and user need: {st.session_state.query_data}.\
                        You try to give user more information for each place. A response shortlist of relevant properties with brief descriptions. Highlights of key features that match the user's preferences (e.g., quiet neighborhood, good schools, park proximity).")
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

    # Generate for advertise an ideal place
    if st.session_state.flag_check == 1:
        system_promt = f"You are an advertised assistant, base on the 4 property information : {st.session_state.retrieval_data}, and user need: {st.session_state.query_data}.\
                        You try to give user more information for each place. Try to please user and highlight of key features that match the user's preferences (e.g., quiet neighborhood, good schools, park proximity).\
                        Help user to find the best place among {st.session_state.retrieval_data}"
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt,system_promt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
