import numpy as np
import random
import traceback
import resource
import sys
import os
import pickle
from dotenv import load_dotenv
import logging
import click
import torch
import time
import re
import json
import networkx as nx
from datetime import datetime
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from classes.dots import Dot, MemoryStream
from classes.utils import (get_lca, count_hypo_docs, extract_entities, create_graph, get_hypo_dots, get_dots_with_docs, save_load_memory_stream, load_documents, make_prompt, load_quantized_model_qptq, load_full_model)
import classes.createUI as Ui

from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
)

sys.setrecursionlimit(10000)

failure_counter = 0
def increment_counter():
    global failure_counter
    failure_counter += 1

if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)


def load_model(model_type="both", device_type="cuda", model_id=MODEL_ID, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library. Default will load openAI

    Args:
        model_type (str): Llama or openAI or both(default)
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        both, none or one of the models
    """
    logging.info(f"Loading model_type: {model_type}")
    local_llm = None
    llm = None

    if model_type in ["llama", "both"]:
        if model_basename is not None:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
        else:
            model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(model_id)

        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.1,
            top_k=40,
            repetition_penalty=1.176,
            generation_config=generation_config,
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)
        logging.info(f"Local LLM Loaded with model_type: {model_type}")
        
    if model_type in ["openAI", "both"]:
        # Loads OpenAI model by default
        llm = AzureChatOpenAI(
            temperature=0.1,
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-05-15",
            azure_deployment="sample_deployment",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        logging.info(f"OpenAI model Loaded with model_type: {model_type}")
    
    return {"local_llm":local_llm, "openAI":llm}

# Post processing script to make the splitting work. depends on what prompt returns.
# Works for dry runs (saves the dots to a JSON file and will reload it if it exists; that way, we don't invoke model with same data again and again)
def break_dots(document, dataset, llm=None, break_type="condensed"):
    """ This calls LLM, get dot strings for this document and return the list of strings """

    if dataset == 'crescent':
        save_dir = os.getcwd() + '/saves/crescent'
    if dataset == 'manpad':
        save_dir = os.getcwd() + '/saves/manpad'
    if dataset == 'atlantic_storm':
        save_dir = os.getcwd() + '/saves/atlantic_storm'

    if break_type == "condensed":
        save_dir = save_dir + "_condensed"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Dry runs, check if file/dots are already saved
    json_files = [file for file in os.listdir(save_dir) if file.endswith('.json')] 
    for json_file in json_files:
        with open(os.path.join(save_dir, json_file), 'r') as f:
            data = json.load(f)
            # Check if the filename in metadata matches the one provided
            if data.get('filename') == document['filename']:
                logging.info(f"Dots for file {format(document['filename'])} loaded from json")
                return data.get('dots')

    # Otherwise, run with LLM
    if llm['openAI']:
        breakdown_prompt = make_prompt(document['content'], prompt_type="break", break_type=break_type)
        # try catching openAI generation in case it fails and we need to fall back to llama
        try:
            resp = llm['openAI'].invoke(breakdown_prompt).content
        except Exception as e:
            logging.error(f"Error: {e} occured in break_dot for document {document['filename']}")
            increment_counter()
            raise e
    
    elif llm['local_llm']:
        breakdown_prompt = make_prompt(document['content'], model_type="llama", prompt_type="break")
        resp = llm['local_llm'].invoke(breakdown_prompt)

    # Printing for debugging
    logging.info(f"Response from LLM: {resp}")
    
    if break_type == "condensed":
        break_dots = [resp.strip().strip()]
    else:
        # postprocessing of the response
        split_resp = resp.strip().split("\n") # Each string now represents one dot
        # print(split_resp)

        pattern = r'^(\*|-|\d+\.)' # Regex pattern

        # Retrieves the dots
        break_dots = []
        for line in split_resp:
            split = line.split() if line.split() else None
            bullet = split[0] if split else None
            if bullet != None:
                match = re.match(pattern, bullet)
                if match != None and match.group() != None:
                    span = match.span()
                    break_dots.append(line[span[1]:].strip())
        # take unique strings from break_dots
        break_dots = list(set(break_dots))

    # Save the document for any future dry runs
    json_data = {
        'filename': document['filename'],
        'date': document['date'],
        'content': document['content'],
        'dots': break_dots
    }
    json_file = os.path.join(save_dir, f"{document['filename']}.json")
    with open(json_file, 'w') as json_file:
        json.dump(json_data, json_file)

    return break_dots

def relevance_response(list_of_dots, query_dot, llm):
    """
    This function will take a list of dots and a query_dot and call the LLM to get the relevant dots
    """
    # create prompt template for merge with this list of dots and get response from LLM
    logging.info(f"Calling LLM driven relevance: with {[x.id for x in list_of_dots]} for query dot: {query_dot.id}")
    relevance_prompt = make_prompt(list_of_dots, prompt_type="relevant", query=query_dot.info)
    
    if llm['openAI']:
        try:
            resp = llm['openAI'].invoke(relevance_prompt)
        except Exception as e:
            logging.error(f"Error: {e} occured in relevance for query dot: {query_dot.id}, Skipping")
            increment_counter()
            return list_of_dots
    else:
        logging.info(f"No relevance testing for local LLM, Skipping")
        return list_of_dots

    # First split the resp.content by lines. And then regex match starting number with [] or starting number followed by dot
    all_lines = resp.content.split("\n")

    # printing for debugging:
    # print(all_lines)

    # Define the regex pattern with a capturing group for the number
    pattern = r"^\[?(\d+)\]?\s*\.?\s+.*"
    relevant_dots_index = []
    # Iterate through sentences and match the pattern
    for sentence in all_lines:
        match = re.match(pattern, sentence)
        if match:
            number = int(match.group(1))  # Access the captured group
            if number >=1 and number <= len(list_of_dots):
                relevant_dots_index.append(number-1)

    # taking set from the index, in case there are duplicate matching
    relevant_dots_index = list(set(relevant_dots_index))
    logging.info(f"Calling LLM driven relevance: Relevant dots index: {relevant_dots_index}")
    list_of_dots = [list_of_dots[i] for i in relevant_dots_index]
    return list_of_dots

def get_response_list_based(list_of_dots, dot, llm, memory_stream, additional_dots, doc_only_hypo, word_limit=100):
    if llm['openAI']:
        model_type = "openAI"
        # make merge prompt, differs based on additional data present or not
        if additional_dots:
            dots_with_doc = get_dots_with_docs(list_of_dots, llm, model_type)
            logging.info(f"List of dots to merge: {[x.id for x in list_of_dots]} for dot: {dot.id}, with additional dots: {[x.id for x in dots_with_doc]}")
            merge_prompt = make_prompt(list_of_dots, model_type=model_type, prompt_type="merge", additional_dots=dots_with_doc, word_limit=word_limit)
        elif doc_only_hypo:
            dots_with_doc = get_dots_with_docs(list_of_dots, llm, model_type, includes_own=True)
            logging.info(f"Doc based dots to merge: {[x.id for x in dots_with_doc]} for dot: {dot.id},")
            merge_prompt = make_prompt(dots_with_doc, model_type=model_type, prompt_type="merge", word_limit=word_limit)
        else:
            logging.info(f"List of dots to merge: {[x.id for x in list_of_dots]} for dot: {dot.id}")
            merge_prompt = make_prompt(list_of_dots, model_type=model_type, prompt_type="merge", word_limit=word_limit)
        
        # try catching openAI generation in case it fails and we need to fall back to llama
        try:
            response = llm['openAI'].invoke(merge_prompt).content.strip()
        except Exception as e:
            logging.error(f"Error: {e} occured in retrieve_merge for dot {dot.id}")
            increment_counter()
            # if openAI fails, call llama, if llama is not available, raise the exception, 
            if llm['local_llm']:
                model_type = "llama"
                if additional_dots:
                    dots_with_doc = get_dots_with_docs(list_of_dots, llm, model_type)
                    logging.info(f"List of dots to merge: {[x.id for x in list_of_dots]} for dot: {dot.id}, with additional dots: {[x.id for x in dots_with_doc]}")
                    merge_prompt = make_prompt(list_of_dots, model_type=model_type, prompt_type="merge", additional_dots=dots_with_doc, word_limit=word_limit)
                elif doc_only_hypo:
                    dots_with_doc = get_dots_with_docs(list_of_dots, llm, model_type, includes_own=True)
                    logging.info(f"Doc based dots to merge: {[x.id for x in dots_with_doc]} for dot: {dot.id},")
                    merge_prompt = make_prompt(dots_with_doc, model_type=model_type, prompt_type="merge", word_limit=word_limit)
                else:
                    logging.info(f"List of dots to merge: {[x.id for x in list_of_dots]} for dot: {dot.id}")
                    merge_prompt = make_prompt(list_of_dots, model_type=model_type, prompt_type="merge", word_limit=word_limit)
                response = llm['local_llm'].invoke(merge_prompt).strip()
            else:
                raise e

    elif llm['local_llm']:
        model_type = "llama"
        if additional_dots:
            dots_with_doc = get_dots_with_docs(list_of_dots, llm, model_type)
            logging.info(f"List of dots to merge: {[x.id for x in list_of_dots]} for dot: {dot.id}, with additional dots: {[x.id for x in dots_with_doc]}")
            merge_prompt = make_prompt(list_of_dots, model_type=model_type, prompt_type="merge", additional_dots=dots_with_doc, word_limit=word_limit)
        elif doc_only_hypo:
            dots_with_doc = get_dots_with_docs(list_of_dots, llm, model_type, includes_own=True)
            logging.info(f"Doc based dots to merge: {[x.id for x in dots_with_doc]} for dot: {dot.id},")
            merge_prompt = make_prompt(dots_with_doc, model_type=model_type, prompt_type="merge", word_limit=word_limit)
        else:
            logging.info(f"List of dots to merge: {[x.id for x in list_of_dots]} for dot: {dot.id}")
            merge_prompt = make_prompt(list_of_dots, model_type=model_type, prompt_type="merge", word_limit=word_limit)
        response = llm['local_llm'].invoke(merge_prompt).strip()

    split_identifiers = ["deduced that "]
    # split the response with the identifiers
    for identifier in split_identifiers:
        if identifier in response:
            response = response.split(identifier)[1].strip()
            break

    # debugging
    logging.info(f"Response from LLM: {response}")

    # Make sure the response is not empty
    if len(response) <= 20:
        logging.info(f"ALERT: Response is very small/empty, skipping creating new hypothesis(merge) for dot: {dot.id}")
        return
    
    return response

def retrieve_merge(dot, llm, memory_stream, relevance, additional_dots, doc_only_hypo, search_type, entity_wise_search=False, merge_hypo_dot=False, word_limit=100):
    """
    This is the main center of operation. Given a dot, it will first call the retrieve from memory stream and get a list of dots to merge and create a new hypothesis
    Filtering of non-parented dots will be taken care in the retrieve() of the memory stream class
    So new merged hypothesis will create new dot, will update all comprising dots' parents, will also add them as children of the hypothesis dot.
    Also create a new dot with dot_string, parent is the new merged dot and add it to the memory stream
    
    Returns: a Hypohtesis dot
    
    Recursive function
    """
    logging.info(f"Merge called with dot: {dot.id}")

    # first search it
    # Get the entities if possible, if it works, it will return entities in string, otherwise the original dot itself
    if entity_wise_search:
        dot_string_to_search = extract_entities(dot, llm)
    else:
        dot_string_to_search = dot
    list_of_dots, filter_by_parents = memory_stream.retrieve(dot_string_to_search)

    # end of the recursive loop, return blank list
    if len(list_of_dots) == 0:
        logging.info(f"No match found for dot: {dot.id}")
        return
    # or else, create hypothesis_dot and call the function again to search for new level of connection
    else: 
        # if relevance is activated
        if relevance:
            list_of_dots = relevance_response(list_of_dots, dot, llm)
        if len(list_of_dots) == 0:
            logging.info(f"No LLM driven relevant match found for dot: {dot.id}")
            return


        # if new search version is activated, need to do some extra things, Then everything is same as before, we can do additional dot or not, doc_only_hypo or not
        # For this case we should pass the dots through LLM relevance first to dial down the number of dots. Can check if it helps or not
        # First must assert if filter_by_parents is false, then call the get_parents function to get a list of dots we can use to merge this dot
        if search_type=="v2":
            assert filter_by_parents == False, "Filter by parents must be False for search version 2"
            # calling dot can appear in the list of dots returned only in recursive calls, when the hypo dot is already parents of doc dots
            list_of_dots = get_hypo_dots(list_of_dots)
            if dot in list_of_dots:
                list_of_dots.remove(dot)
            if len(list_of_dots) == 0:
                logging.info(f"No match found for dot in v2: {dot.id}")
                return
        elif search_type=="v3":
            assert additional_dots == False and doc_only_hypo == True and merge_hypo_dot==False, "Additional dots, doc_only_hypo and merge_hypo_dot must be False for search version 3"
            # calling dot can appear in the list of dots returned only in recursive calls, when the hypo dot is already parents of doc dots
            list_of_dots = get_lca(memory_stream, list_of_dots)
            if dot in list_of_dots:
                list_of_dots.remove(dot)
            if len(list_of_dots) == 0:
                logging.info(f"No match found for dot in v3: {dot.id}")
                return

        # create prompt template for merge with this list of dots and get response from LLM
        logging.info(f"Retrieved final dots {[x.id for x in list_of_dots]} for dot: {dot.id}")
        
    
        # added the calling dot to list of children
        list_of_dots.append(dot)


        # We will update the tree first and then update the information inside the node
        # first count how many hypo dot is in there
        hypo_dot, doc_dot = count_hypo_docs(list_of_dots)
        
        if len(hypo_dot) == 0:
            # no hypo dot means need to create one with all the list_of_dots
            hypothesis_dot = Dot("", children_dots=list_of_dots)
            logging.info(f"Hypothesis dot created, id: {hypothesis_dot.id}, with children dots: {[x.id for x in list_of_dots]}")

            # update the parents of this list of dots+ the dot the function got called with
            for item in list_of_dots:
                logging.info(f"adding {hypothesis_dot.id} for dot:{item.id} as parent, vice-versa")
                item.update_parents(hypothesis_dot)
        
        # if merge_hypo_dot is activated, we will try to update the existing hypothesis dot, otherwise we will create a new one
        elif len(hypo_dot) == 1:
            hypothesis_dot = hypo_dot[0]
            logging.info(f"Hypothesis dot found, id: {hypothesis_dot.id}, with children dots: {[x.id for x in list_of_dots]}")
            
            for item in doc_dot:
                logging.info(f"adding {hypothesis_dot.id} for dot:{item.id} as parent, vice-versa")
                hypothesis_dot.update_children(item)
                item.update_parents(hypothesis_dot)
        
        # this case is when there are two or more hypothesis dots, we first get the topmost parents dots, then merge everything into one hypothesis dot
        else:
            # get the topmost parents
            logging.info(f"Multiple hypothesis dots found, id: {[x.id for x in hypo_dot]}")
            list_of_dots = get_hypo_dots(list_of_dots)
            new_hypo_dot, new_doc_dot = count_hypo_docs(list_of_dots)
            if len(new_hypo_dot) >= 2: 
                # we will merge all the hypothesis dots into one
                hypothesis_dot = Dot("", children_dots=list_of_dots)
                logging.info(f"Hypothesis dot created, id: {hypothesis_dot.id}, with children dots: {[x.id for x in list_of_dots]}")

                # update the parents of this list of dots+ the dot the function got called with
                for item in list_of_dots:
                    logging.info(f"adding {hypothesis_dot.id} for dot:{item.id} as parent")
                    item.update_parents(hypothesis_dot)
            elif len(new_hypo_dot) == 1:
                logging.error(f"Multiple hypothesis dots found at first but get_hypo returned less than 2, id: {[x.id for x in new_hypo_dot]}")
                hypothesis_dot = new_hypo_dot[0]
            else:
                raise Exception("No hypothesis dot found after merging multiple hypothesis dots")
        
        response = get_response_list_based(hypothesis_dot.children_dots, dot, llm, memory_stream, additional_dots=additional_dots, doc_only_hypo=doc_only_hypo, word_limit=100)
        hypothesis_dot.update_info(response)

    
        # at the end add hypothesis dots to the memory stream
        memory_stream.newDots([hypothesis_dot]) 
        logging.info(f"Hypothesis dot {hypothesis_dot.id} added to memory stream ")
        return


def build_memory_stream(documents, dataset, llm, memory_stream, output_path, relevance=True, additional_dots=True, doc_only_hypo=True, filter_by_parents=True, search_type="v2", break_type="condensed", entity_wise_search=False, merge_hypo_dot=False, tag="", word_limit=100):
    for document in documents:
        logging.info(f"Running document: {document['filename']}")
        try:
            dots = break_dots(document, dataset, llm, break_type)
        except Exception as e:
            logging.error(f"Error: {e} occured in break_dot for file: {document['filename']}. Skipping this document")
            # logging.error(traceback.format_exc())
            increment_counter()
            continue

        for dot_string in dots:
            # make loop wait for user response
            # input("Press Enter to continue...")
            logging.info(f"Running dot_string: {dot_string} /// document: {document['filename']}")
            # create a new dot with this dot string
            new_dot = Dot(dot_string, doc=document['filename'])

            # Try catch block in case LLM invokation inside retrieve_merge pickes up any content moderation
            # Will change it to LLM.invoke locations when local model will be added
            try:
                # call the recursive function to create chains of hypotheses
                retrieve_merge(new_dot, llm, memory_stream, relevance=relevance, additional_dots=additional_dots, doc_only_hypo=doc_only_hypo, search_type=search_type, entity_wise_search=entity_wise_search, merge_hypo_dot=merge_hypo_dot, word_limit=word_limit)
            except Exception as e:
                logging.error(f"Error: {e} occured in retrieve_merge for dot: {new_dot.id}")
                logging.error(traceback.format_exc())
                increment_counter()
                if "list index out of range" in str(e):
                    logging.info("List index out of range error occured, stopping the process")
                    logging.info(f"Processing ended for {len(documents)} document : Total dots: {len(memory_stream.dots)}, Failure: {failure_counter}")
                    filename = f"type-{search_type}_dataset-{dataset}_relevance-{relevance}_parent-{filter_by_parents}_additional-{additional_dots}_doc-{doc_only_hypo}"
                    save_load_memory_stream(os.path.join(output_path, filename+".json"), mode="save", memory_stream=memory_stream)
                    exit()
                pass

            # in any case, we need to keep track of this dot in memorystream
            # When it returns, add this dot to memorystream
            memory_stream.newDots([new_dot])
            # save_load_memory_stream(os.path.join(output_path, str(new_dot.id)+".json"), mode="save", memory_stream=memory_stream)
        logging.info(f"Processing ended for document: {document['filename']}")
        # save_load_memory_stream(os.path.join(output_path, str(document['filename'])+".json"), mode="save", memory_stream=memory_stream)
        # save_load_memory_stream(os.path.join(output_path, str(document['filename'])+".pickle"), mode="save", memory_stream=memory_stream)

    # saving the final memory stream to a both pickle and json file
    try:
        logging.info(f"Processing ended for {len(documents)} document : Total dots: {len(memory_stream.dots)}, Failure: {failure_counter}")
        filename = f"Subplot-{tag}_type-{search_type}_dataset-{dataset}_relevance-{relevance}_parent-{filter_by_parents}_additional-{additional_dots}_doc-{doc_only_hypo}"
        save_load_memory_stream(os.path.join(output_path, filename+".json"), mode="save", memory_stream=memory_stream)
        save_load_memory_stream(os.path.join(output_path, filename+".pickle"), mode="save", memory_stream=memory_stream)
    except Exception as e:
        logging.error(f"Error: {e} occured in saving memory stream")
        pass

    # make a graph for the UI
    try:
        graph = create_graph(memory_stream.dots)
        # with open(os.path.join(output_path, "graph.gpickle"), 'wb') as pfile:
        #     pickle.dump(graph, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        # create UI
        # also making the html file for UI
        ui_file_name = f"{dataset}_{tag}.html" 
        Ui.create_UI(graph, os.path.join(output_path, ui_file_name), False)
    except Exception as e:
        logging.error(f"Error: {e} occured in creating graph")
        pass


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--model_type",
    default="both",
    type=click.Choice(
        [
            "llama", 
            "openAI",
            "both"
        ],
    ),
    help="Model type (Default is both). Mention type (openAI, llama or both) to automatically load the right model and choose the right prompt template",
)
@click.option(
    "--dataset",
    "-s",
    default="crescent",
    type=click.Choice(
        [
            "crescent", 
            "manpad",
            "atlantic_storm"
        ],
    ),
    help="Dataset. Default is Crescent",
)
@click.option("--output_path","-o",default='results/',help="Output path")
@click.option("--data_path", "-d", default='data', help="Dataset path")
@click.option('--relevance/--no-relevance', default=False, help="Deactivate relevance search through LLM")
@click.option("--filter_by_parents/--no-filter_by_parents", default=False, help="Whether to filter the relevancy by dot's with parents, default is True")
@click.option('--additional_dots/--no-additional_dots', default=False, help="Add additional dots to merge prompt(leaf dots), default is True")
@click.option('--doc_only_hypo/--no-doc_only_hypo', default=True, help="Whether to consider only documents' dots to generate hypothesis, default is true")
@click.option('--search_type', default="v3", type=click.Choice(["v3"],), help="")
@click.option('--break_type', default="condensed", type=click.Choice(["sentence","condensed"],), help="which search type, v1 or v2. Readme files has details about it")
@click.option('--entity_wise_search/--no-entity_wise_search', default=False, help="Enable or disable entity based search, this is the outer layer one")
@click.option('--merge_hypo_dot/--no-merge_hypo_dot', default=False, help="Enable or disable updating current hypothesis dot")
def main(device_type, model_type, dataset, data_path, output_path, relevance, filter_by_parents, additional_dots, doc_only_hypo, search_type, break_type, entity_wise_search, merge_hypo_dot):
    """
    Read the dataset and call the required classes to build up the datasets.

    """

    # take the version input from the user
    tag = input("Enter an unique tag for this run: ")

    # check if models directory do not exist, only for llama model
    if model_type in ["llama", "both"] and not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    # check if output_path exist and makeone if it doesnot
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    # OpenAI model by default if we do not provide any arguments
    llm = load_model(model_type=model_type)
    # llm = load_model(model_type="openAI", model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    # Create embeddings model
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

    # load documents
    documents = load_documents(data_path, dataset)  # List of docs in string format
    print(len(documents), documents[0])
    # for item in documents:
    #     break_dots(item, dataset, llm, break_type)

    # Initialize memory stream object, which creates a vector database
    memory_stream = MemoryStream(
        embeddings=embeddings,
        client_settings=CHROMA_SETTINGS,
        filter_by_parents=filter_by_parents,
        llm = llm
    )

    build_memory_stream(documents, dataset, llm, memory_stream, output_path, relevance=relevance, additional_dots=additional_dots, doc_only_hypo=doc_only_hypo, filter_by_parents=filter_by_parents, search_type=search_type, break_type=break_type, entity_wise_search=entity_wise_search, merge_hypo_dot=merge_hypo_dot, tag=tag, word_limit = 100)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO, handlers=[
        logging.FileHandler(os.path.join("results/run.log")),
        logging.StreamHandler()
        ]
    )
    main()
