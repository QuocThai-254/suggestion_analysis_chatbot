# In-depth property analysis and suggestion chatbot

# Table of content
1. [About](#about)
2. [Set up Environment](#set-up-environment)
3. [How to run](#how-to-run)

## About

This is the code that create a in-depth property analysis and suggestion chatbot with provided dataset. For each input prompt from user, **sentiment analysis** is applied to collect the data and adapt future suggestions accordingly. For more detail about the pipline and explantion, please find it in the [pdf folder](./pdf/).

## Set up Environment

1. Set up the environment with python=3.9 or with conda:
```
conda create -n be_earning_chat python=3.9
```
2. Install requirements packages by running the cli:

```
pip install -e .
```

Then the environment for the chatbot was already setup.

## How to run

1. Before running chatbot, we need to provide it the data for retrival and suggestion task. Since I already set up the default for arguments:

```
python .\prepare_data.py
# For more details or arguments information
python .\prepare_data.py -h
```

2. Chatbot can be used via the interface provided by **Streamlit** at local host. Since when I do the code test, I run to test so much time that I reach the limit time for **Replicate** :'( . Therefore, please go to the main file in lines *174, and 175* and change the *Replicate API token* of yours to run the chatbot.

In file main.py file, change here:

```
os.environ['REPLICATE_API_TOKEN'] = "r8_***********************************"
replicate_api = "r8_***********************************"
```

Then run the cli:

```
streamlit run main.py
```

