# for helper utils
import os
import pandas as pd
import pickle
import re
import json
import networkx as nx
import logging
from datetime import datetime
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from classes.dots import Dot, MemoryStream
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# for loading models
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from constants import MODELS_PATH, PROMPT_PATH

class Entity(BaseModel):
    """Output the LLM input with the entities in the search query."""
    entities: str = Field(description=("All entities without commas"))

def extract_entities(dot, llm):
    # fixing prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at extracting entities from the sentence. All entities must be from the sentence."),
            ("human", "{question}"),
        ]
    )
    if isinstance(dot, str):
        question = dot.strip()
    else:
        question = dot.info.strip()

    # We will do this with llama only for now
    if llm['openAI']:
        structured_llm = llm['openAI'].with_structured_output(Entity)
        query_analyzer = prompt | structured_llm
        try:
            return query_analyzer.invoke({"question": question}).entities.strip()
        except Exception as e:
            logging.error(f"Error: {e} occured in entity_extraction for dot {dot}")
            return dot
    else:
        return dot


def root_node(nx_graph):
    all_comps = list(nx.weakly_connected_components(nx_graph))
    largest_comp = max(all_comps, key=len)
    for node in largest_comp:
        if node.parents == []:
            return node

def final_report(llm, nx_graph):
    root = root_node(nx_graph)
    children_dots = root.children_dots
    if llm['openAI']:
        merge_prompt = make_prompt(children_dots, prompt_type="final")
        # try catching openAI generation in case it fails and we need to fall back to llama
        try:
            response = llm['openAI'].invoke(merge_prompt).content.strip()
            return response
        except Exception as e:
            logging.error(f"Error: {e} occured in final_report")
            pass
    raise Exception("No LLM found to generate final report")
    
    
def count_hypo_docs(list_of_dots):
    doc_dots = []
    hypo_dots = []
    for dot in list_of_dots:
        if dot.doc:
            doc_dots.append(dot)
        else:
            hypo_dots.append(dot)
    return hypo_dots, doc_dots

def create_graph(list_of_dots):
    """create the tree with this list_of_dots"""
    G = nx.DiGraph()

    ordered_list = sorted(list_of_dots, key=lambda x: x.id)
    # Add nodes to the graph
    G.add_nodes_from(ordered_list)

    # Add edges to the graph based on relationships
    for dot in ordered_list:
        for parent in dot.parents:
            G.add_edge(dot, parent)
    
    return G

# make a graph first and then identify the connected components
def get_lca(memory_stream, list_of_dots):
    """get the lowest common ancestor of the list of dots"""
    # create the graph
    G = create_graph(memory_stream.dots)
    # get the connected components
    connected_components = list(nx.weakly_connected_components(G))
    # filter out the connected components that have the any of the list_of_dots
    selected_component = []
    for component in connected_components:
        if any(x in component for x in list_of_dots):
            selected_component.append(component)
    
    # we will iterate with each component and find the lca for items of list_of_dots in that component
    lca = []
    for component in selected_component:
        # identify the items in the list_of_dots that are in this component
        dots_in_component = [x for x in list_of_dots if x in component]
        # get the lca for this component
        ancestors_sets = [set(nx.descendants(G, node)) for node in dots_in_component]
        # check if every element of ancestor set if empty. In that case it is only a document node
        if all([len(x) == 0 for x in ancestors_sets]):
            lca.extend(dots_in_component)
            continue
        # Find the common ancestors by taking the intersection of all sets otherwise
        common_ancestors = sorted(list(set.intersection(*ancestors_sets)), key=lambda x: x.id)
        lca.append(common_ancestors[0])
    
    return lca


# This is not so straightforward like the descendants, as we can get parents multiple time from different children
# Cannot just extend our way out
def get_hypo_dots(list_of_dots):
    """isolate the dots without parents"""
    dots_without_parents = []
    for dot in list_of_dots:
        topmost_parent = get_topmost_parent(dot)
        if topmost_parent not in dots_without_parents:
            dots_without_parents.append(topmost_parent)
    return dots_without_parents

# get topmost parent of a dot
def get_topmost_parent(dot):
    if not dot.parents:
        return dot
    else:
        return get_topmost_parent(dot.parents[0])

# new function to see if there are upper level hypo dots we can use for merging
def get_upper_hypo_dots(list_of_dots, filter_by_parents=True):
    new_dots_with_docs = []
    for item in list_of_dots:
        if not item.parents or any([len(x.info) == 0 for x in item.parents]):
            new_dots_with_docs.append(item)
        else:
            new_dots_with_docs.extend(item.parents)
    new_dots_with_docs = list(set(new_dots_with_docs)) # remove duplicates

    if not filter_by_parents:
        return new_dots_with_docs

    # need to check if any of the dots are already parents of other dots
    logging.error(f"UTIL: Filtering by parents ON!!!")
    parent_filtered_dots = []
    for item in new_dots_with_docs:
        if not any([x in new_dots_with_docs for x in item.parents]):
            parent_filtered_dots.append(item)

    return parent_filtered_dots

def validation_dots_with_doc(dots_with_doc, llm, model_type, filter_by_parents=False):
    merge_prompt = make_prompt(dots_with_doc, model_type=model_type, prompt_type="merge")
    valid = False
    if model_type == "openAI":
        num_of_tokens = llm['openAI'].get_num_tokens_from_messages(merge_prompt)
        valid = True if num_of_tokens < 3500 else False
    else:
        num_of_tokens = llm['local_llm'].get_num_tokens(merge_prompt)
        valid = True if num_of_tokens < 3000 else False
    
    # condition to exit the validation
    # logging.info(f"UTIL: num_of_tokens: {num_of_tokens}")
    if valid:
        return dots_with_doc
    else:
        # making notes that we had to enter to the validation which indicates that there are too many tokens
        logging.info(f"UTIL: Entering validation for too many tokens.")
        new_dots_with_docs = []
        for item in dots_with_doc:
            if not item.parents or any([len(x.info) == 0 for x in item.parents]):
                new_dots_with_docs.append(item)
            else:
                new_dots_with_docs.extend(item.parents)
        new_dots_with_docs = list(set(new_dots_with_docs)) # remove duplicates

        if not filter_by_parents:
            return validation_dots_with_doc(new_dots_with_docs, llm, model_type)

        # this is to remove the dots that are already parents of other dots
        logging.error(f"UTIL: Filtering by parents ON!!!")
        parent_filtered_dots = []
        for item in new_dots_with_docs:
            if not any([x in new_dots_with_docs for x in item.parents]):
                parent_filtered_dots.append(item)

        return validation_dots_with_doc(parent_filtered_dots, llm, model_type)
        
def get_dots_with_docs(list_of_dots, llm, model_type, includes_own=False, validation=True):
    """isolate the dots with documents"""
    dots_with_doc = []
    for dot in list_of_dots:
        descendants = get_descendants(dot)
        if includes_own:
            descendants.append(dot)
        dots_with_doc.extend([x for x in descendants if x.doc != None])

    # check if the number of dots with doc is under limit
    # logging.info(f"UTIL: BEFORE: dots with documents: {[x.id for x in dots_with_doc]}")
    if validation:
        new_dots_with_doc = validation_dots_with_doc(dots_with_doc, llm, model_type)
        return new_dots_with_doc
    # logging.info(f"UTIL: AFTER: dots with documents: {[x.id for x in new_dots_with_doc]}")
    return dots_with_doc


# changing it to a post-order traversal, keeping the functionality same
def get_descendants(dot, exclude_self=True):
    descendants = []
    for child in dot.children_dots:
        descendants.extend(get_descendants(child, exclude_self=False))
    if not exclude_self:
        descendants.append(dot)
    return descendants

# also supports taking in a list_of_dots for pickling, not changing the argument's name though
def save_load_memory_stream(file_name, mode="save", memory_stream=None):
    """mention mode "save" or "load". It will only work with the list of dots"""
    if "json" not in file_name:
        if mode=="save":
            with open(file_name, "wb") as fp:   #Pickling
                if isinstance(memory_stream, list):
                    pickle.dump(memory_stream, fp)
                else:
                    pickle.dump(memory_stream.dots, fp)
        else:
            with open(file_name, "rb") as fp:   # Unpickling
                return pickle.load(fp)
    elif "json" in file_name:
        if mode=="save":
            with open(file_name, 'w', encoding='utf-8') as fp:
                json.dump(memory_stream.snapshot(), fp, ensure_ascii=False, indent=4)
        else:
            with open(file_name, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                return data

def load_atlantic_storm_dataset(input_directory):
    #load the dataframe from the csv file
    df = pd.read_csv("saves/atlantic_storm.csv")
    # make the serial column the new index
    df = df.sort_values(by=['serial'])
    df.set_index('serial', inplace=True)
    df = df.drop(columns=["file_content"])
    
    docs = []
    # iterate through the dataframe 
    for index, row in df.iterrows():
        report_date = row['Date']
        filename = row['file_name']
        input_file_path = os.path.join(input_directory, filename)

        if os.path.isfile(input_file_path):
            # Read the content of the file
            with open(input_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                docs.append({'date':report_date, 'filename':filename, 'content':file_content.strip()})

    return docs

# Loads documents, sorting them by date. Read the date from inside the file and sort
def load_documents(input_directory, dataset):
    # Ensure the input directory exists
    if not os.path.exists(input_directory):
        print(f"The input directory '{input_directory}' does not exist.")
        return

    # Atlantic storm needs different load mechanism
    if dataset == 'atlantic_storm':
        return load_atlantic_storm_dataset(input_directory)
    
    # List all files in the input directory
    files = os.listdir(input_directory)

    # Regex pattern for date extraction
    if dataset == 'crescent': # Crescent dataset
        date_pattern = r'^(Report Date:|Report Date|Report dated:|Report dated|Report date|Report date:)\s*(\d+ \w+, \d+).'
    if dataset == 'manpad': #Manpad dataset
        date_pattern = r'^(doc_\d+)\s*\[(\d+ \w+, \d+)\]'

    # Dictionary to map month names to their numeric representation
    month = {
        "January": "01",
        "February": "02",
        "March": "03",
        "April": "04",
        "May": "05",
        "June": "06",
        "July": "07",
        "August": "08",
        "September": "09",
        "October": "10",
        "November": "11",
        "December": "12"
    }

    docs = []
    for filename in files:
        input_file_path = os.path.join(input_directory, filename)

        if os.path.isfile(input_file_path):
            # Read the content of the file
            with open(input_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                # Find the line that contains the report date
                for line in file_content.split('\n'):
                    match = re.match(date_pattern, line)
                    if match != None and match.group() != None:
                        date_string = match.group(2)
                        # Convert the date string to a datetime object
                        for month_name, month_number in month.items():
                            if month_name in date_string:
                                date_string = date_string.replace(month_name, month_number)
                                break
                        report_date = datetime.strptime(date_string, '%d %m, %Y').strftime('%Y%m%d')
                        index = line.find(":", match.end()) + 1 if line.find(":", match.end()) != -1 else match.end()
                        docs.append({'date':report_date, 'filename':filename, 'content':line[index:].strip()})

    # Sort documents by date
    return sorted(docs, key=lambda x: x['date'])

# Changing it to include a basic llama prompt and retained only one type of prompt
def merge_prompt_type(documents, model_type="openAI", word_limit=100, merge_type="default"):
    logging.info("ALERT: Using the merge prompt type function")
    # validate the documents that there is not a mixed of hypo and doc dots>2
    hypo_dots, doc_dots = count_hypo_docs(documents)
    if len(hypo_dots) >=1 and len(doc_dots) >= 2:
        raise Exception("Cannot merge hypo and 2 doc dots in the same prompt")
    if model_type != "openAI":
        if isinstance(documents[0], str):
            doc_string_full = "\n".join(documents)
        else:
            doc_string_full = "\n".join([x.info for x in documents])
        system_prompt = "You are a helpful assistant"
        if merge_type == "person":
            template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nUser: List of evidences:\n{document}\n\nQuestion: What is the connection between persons from the above chain of evidences? Think about aliases, numbers and addresses. Explain in three sentences, include relevant information.\n\nAnswer:\n[/INST]"
        else:
            template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nUser: List of evidences:\n{document}\n\nQuestion: What can you deduce from the above chain of evidences? Think about all the names, aliases, numbers and addresses. Synthesize in three sentences, include relevant information.\n\nAnswer:\n[/INST]"
        prompt_template = PromptTemplate(
            input_variables=["system_prompt", "document"],
            template=template,
        )
        return prompt_template.format(document=doc_string_full, system_prompt=system_prompt)
    else:
        doc_dot_string_full = "\n".join([x.info for x in documents])
        prompt_template = ChatPromptTemplate.from_messages(
            [
                # ("system", "You are an FBI agent, working with field reports. What can you deduce from the following list of reports describing different persons? Look for connection through numbers, aliases, addresses and events. Retain important information from all reports and explain the connection within {word_limit} words."),
                ("system", "You are an FBI agent, working with fictional field reports. What can you deduce from the following list of fictional reports? Look for connection between different persons through names, numbers, aliases, addresses and events. Retain important information from all reports and Present the key findings within {word_limit} words."),
                ("human", "List of reports:\n{doc_dot}"),
            ],
        )
        return prompt_template.format_messages(doc_dot=doc_dot_string_full, word_limit=word_limit)


# We will keep it intact for now but we will write a new revamped prompting function
def make_prompt(documents, prompt_type="break", model_type="openAI", query=None, additional_dots=None, break_type="condensed", word_limit=100, merge_type="default"):
    """
    Will create a prompt with the supplied model_type, prompt_type and the documents and return that, default will do break and openAI
    """
    if model_type == "openAI":
        if prompt_type=="break" and isinstance(documents, str):
            if break_type == "condensed":
                system_prompt = "You are an FBI agent, working with field reports. Condense the reports into fewer sentences. You must include all the names, aliases, numbers and addresses. Make sure the condensed report is within 50 words. If it is too large, break down into chunks."
            elif break_type == "person":
                system_prompt = "Please analyze the report and provide a detailed breakdown of each named human. List each person into separate sentences, including every relevant details. Add ### between each breakdown.\n"
            else:
                system_prompt = "Rewrite the following report such that each sentence includes all entity names it refers to. Break down into excerpts and print in a numbered list.\n"
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "Report:\nReport: {document}"),
                ],
            )
            return prompt_template.format_messages(document=documents)
        
        elif prompt_type=="merge" and isinstance(documents, list):

            # for now use the merge_prompt_type function, which is for hypo_merge cases
            # return merge_prompt_type(documents, model_type=model_type, word_limit=word_limit)
            logging.info("UTIL: Not using the merge prompt type function")

            if isinstance(documents[0], str):
                doc_string_full = "\n".join(documents)
            else:
                doc_string_full = "\n".join([x.info for x in documents])
            if not additional_dots:
                # TODO: streamline this process, remove additional_dots and use only one variable for the merge_type
                if merge_type == "person":
                    prompt_template = ChatPromptTemplate.from_messages(
                        [
                            # ("system", "Act as an intelligence analyst. What can you deduce from the following list of evidence? Synthesize in two sentence. Answer without any preamble."),
                            ("system", "You are an FBI agent, working with field reports. What can you deduce from the following list of reports describing different persons? Look for connection through numbers, aliases, addresses and events. Retain important information from all reports and explain the connection within {word_limit} words."),
                            ("human", "List of fictional reports:\n{document}"),
                        ],
                    )
                    return prompt_template.format_messages(document=doc_string_full, word_limit=word_limit)
                else:
                    prompt_template = ChatPromptTemplate.from_messages(
                        [
                            # ("system", "Act as an intelligence analyst. What can you deduce from the following list of evidence? Synthesize in two sentence. Answer without any preamble."),
                            ("system", "You are an FBI agent, working with fictional field reports. What can you deduce from the following list of fictional reports? Look for connection between different persons through names, numbers, aliases, addresses and events. Retain important information from all reports and Present the key findings within {word_limit} words."),
                            ("human", "List of fictional reports:\n{document}"),
                        ],
                    )
                    return prompt_template.format_messages(document=doc_string_full, word_limit=word_limit)
            else:
                additional_dots = "\n".join([x.info for x in additional_dots])
                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", "Act as an intelligence analyst. What can you deduce from the following list of evidence? Also consider the additional information when making your case. Synthesize in two sentence. Answer without any preamble."),
                        ("human", "List of evidences:\n{document}\nAdditional information:\n{additional_dots}"),
                    ],
                )
                return prompt_template.format_messages(document=doc_string_full, additional_dots=additional_dots)

        elif prompt_type=="relevant" and isinstance(documents, list):
            if isinstance(documents[0], str):
                doc_string_full = "\n".join([f"{i+1}. {x}" for i, x in enumerate(documents)])
            else:
                doc_string_full = "\n".join([f"{i+1}. {x.info}" for i, x in enumerate(documents)])
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "Act like a search engine. You will be given an initial query and a list of sentences. Expand the query with relevant sentences and find out all relevant sentences from the list. You must output the indices of the final list sentences in <code></code> block."),
                    ("human", "Query: {query}\n\nList of Sentences:\n{doc_string_full}"),
                ],
            )
            return prompt_template.format_messages(doc_string_full=doc_string_full, query=query)
        
        elif prompt_type=="final":
            if isinstance(documents[0], str):
                doc_string_full = "\nReport: ".join(documents)
            else:
                doc_string_full = "\nReport: ".join([x.info for x in documents])
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    # ("system", "Act as an intelligence analyst. What can you deduce from the following list of evidence? Synthesize in two sentence. Answer without any preamble."),
                    ("system", "You are an FBI agent, working with field reports. Make a report from these initial reports. Retain all important information from all reports."),
                    ("human", "List of reports:\nReport: {document}"),
                ],
            )
            return prompt_template.format_messages(document=doc_string_full, word_limit=word_limit)
        
        else:
            raise Exception("Prompt type and object type did not match. Break should get a single document as string, merge should get a list of dots")
    
    else:
        if prompt_type=="break" and isinstance(documents, str):
            # TODO: improve break_dot prompt for llama models, this one seems to work fine for now
            # template = "### HUMAN:\nRewrite the following report such that each sentence is standalone and informative.\nReport:\n{document}\n\n\n### RESPONSE: Here is the list:\n"
            template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nUser: {document}\n\nRewrite the above report such that each sentence is standalone and informative.\n\nAnswer: Here is the list of sentences with all entities they refer to:\n[/INST]"
            prompt_template = PromptTemplate(
                input_variables=["system_prompt", "document"],
                template=template,
            )
            return prompt_template.format(document=documents, system_prompt="You are a helpful assistant")
        
        elif prompt_type=="merge" and isinstance(documents, list):
            if isinstance(documents[0], str):
                doc_string_full = "\n".join(documents)
            else:
                doc_string_full = "\n".join([x.info for x in documents])
            system_prompt = "You are a helpful assistant"

            if not additional_dots:
                # TODO: Test and find a better prompt for llama. Also streamline this process, remove additional_dots and use only one variable for the merge_type
                if merge_type == "person":
                    template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nUser: List of evidences:\n{document}\n\nQuestion: What is the connection between persons from the above chain of evidences? Think about aliases, numbers and addresses. Explain in three sentences, include relevant information.\n\nAnswer:\n[/INST]"
                else:
                    template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nUser: List of evidences:\n{document}\n\nQuestion: What can you deduce from the above chain of evidences? Think about all the names, aliases, numbers and addresses. Synthesize in three sentences, include relevant information.\n\nAnswer:\n[/INST]"
                prompt_template = PromptTemplate(
                    input_variables=["system_prompt", "document"],
                    template=template,
                )
                return prompt_template.format(document=doc_string_full, system_prompt=system_prompt)
            
            else:
                additional_dots = "\n".join([x.info for x in additional_dots])

                template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nUser: List of evidences:\n{document}\n\nAdditional information:\n{additional_dots}\n\nQuestion: What can you deduce from the above list of evidences? Also consider the additional information. Synthesize in one sentences.\n\nAnswer:\n[/INST]"
                prompt_template = PromptTemplate(
                    input_variables=["system_prompt", "document", "additional_dots"],
                    template=template,
                )
                return prompt_template.format(document=doc_string_full, additional_dots=additional_dots, system_prompt=system_prompt)

        else:
            raise Exception("Prompt type and object type did not match. Break should get a single document as string, merge should get a list of dots, Local LLM does not support relevance search")


def load_quantized_model_qptq(model_id, model_basename, device_type, logging):

    # The code supports all huggingface models that ends with GPTQ and have some variation
    # of .no-act.order or .safetensors in their HF repo.
    logging.info("Using AutoGPTQForCausalLM for quantized models")

    if ".safetensors" in model_basename:
        # Remove the ".safetensors" ending if present
        model_basename = model_basename.replace(".safetensors", "")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    logging.info("Tokenizer loaded")

    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device_map="auto",
        use_triton=False,
        quantize_config=None,
    )
    return model, tokenizer


def load_full_model(model_id, model_basename, device_type, logging):

    if device_type.lower() in ["mps", "cpu"]:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id, cache_dir="./models/")
        model = LlamaForCausalLM.from_pretrained(model_id, cache_dir="./models/")
    else:
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
        logging.info("Tokenizer loaded")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=MODELS_PATH,
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16,
            # trust_remote_code=True, # set these if you are using NVIDIA GPU
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    return model, tokenizer