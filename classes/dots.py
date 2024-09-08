from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate
import logging
from fuzzywuzzy import fuzz
import re

def split_results(response):
    split_resp = response.strip().split("\n") # Each string now represents one dot
    pattern = r'^(\*|-|\d+\.)' # Regex pattern
    # Retrieves the dots
    entities = []
    for line in split_resp:
        split = line.split() if line.split() else None
        bullet = split[0] if split else None
        if bullet != None:
            match = re.match(pattern, bullet)
            if match != None and match.group() != None:
                span = match.span()
                entities.append(line[span[1]:].strip())
    # take unique strings from break_dots
    entities = list(set(entities))
    entities = [x for x in entities if len(x) > 5]
    return entities

def find_entities(dot, llm):
    template = "<s>[INST] <<SYS>>\nYou are given a fictional report. Extract the entities and print only the extracted phrases in separate lines. Think about the names, number and addresses used.\n<</SYS>>\n\nFictional report: {question}\nSure! Here are the extracted phrases from the given text: [/INST]"
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    if isinstance(dot, str):
        question = dot
    else:
        question = dot.info
    text = prompt_template.format(question=question)
    response = llm['local_llm'].invoke(text)
    return split_results(response.strip())

class Dot:
    """
    In this new design, dots are the only building/memory block
    Dots can be evidential(actual info) or hypothetical(IT)
    Dot will not be updated later, Merge or update will create new one
    so Deduce will stay outside of the class in a global function
    """

    next_id = 1
    def __init__(self, info, doc=None, children_dots=[], parents=[], id=None):
        # info is both dot and synthesized info, need to update when adding new one
        self.info = info
        # doc can be None, for IT type dots
        self.doc = doc
        # This is the list of dots that build this dot, if hypothetical(IT type)
        self.children_dots = []
        self.children_dots.extend(children_dots) # add children if any given during initilization
        # update parents, if this dot get synthesized into another dot, initially empty
        self.parents = []
        self.parents.extend(parents) # add parents if any given during initilization
        # assign unique id to this dot
        if not id:
            self.id = Dot.next_id
            Dot.next_id += 1
        else:
            self.id = id

    # return the content of this dots for printing and logging
    # can also create a dictionary and return that, default
    def __str__(self):
        # return f"<Dot:{self.info}; From Doc:{self.doc}; Belong to IT:{self.ITlist}>"
        return str({'id':self.id, 'info':self.info, 'doc':self.doc, 'parents':str(self.parents)})
    
    # return the content of this dots for printing and logging
    # can also create a dictionary and return that, default
    def get_dict(self):
        # return f"<Dot:{self.info}; From Doc:{self.doc}; Belong to IT:{self.ITlist}>"
        return {'id':self.id, 'info':self.info, 'doc':self.doc, 'parents':[x.id for x in self.parents], 'children_dots':[x.id for x in self.children_dots]}

    # Two dots are equal if they contain the same String (info) and refer
    # to the same document (doc).
    def equals(self, dot):
        return (dot.info == self.info) and (dot.doc == self.doc) and (dot.children_dots==self.children_dots) and (dot.parents==self.parents)
    
    def update_parents(self, hypothesis_dot):
        parents_id_list = [x.id for x in self.parents]
        if hypothesis_dot.id not in parents_id_list and hypothesis_dot.id != self.id:
            self.parents.append(hypothesis_dot)
    
    def update_children(self, synthesized_dot):
        children_id_list = [x.id for x in self.children_dots]
        if synthesized_dot.id not in children_id_list and synthesized_dot.id != self.id:
            self.children_dots.append(synthesized_dot)
    
    def update_info(self, new_info):
        self.info = new_info
    
    
    
class MemoryStream:
    """"
    This is the container for all the dots
    """
    # saving an id for each memory stream DB
    next_id = 1

    def __init__(self, embeddings, client_settings, filter_by_parents=True, llm=None):
        self.dots = []
        self.db = None
        self.embeddings = embeddings
        self.id = MemoryStream.next_id
        MemoryStream.next_id += 1
        self.db = Chroma("langchain_"+str(self.id), embeddings, client_settings=client_settings)
        self.retriever = self.db.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={
                "score_threshold": 0.5,
                "k": 10
                }
        )
        self.filter_by_parents = filter_by_parents
        self.llm = None
        self.llm = llm
        if self.llm and llm['openAI']:
            self._filter = LLMChainFilter.from_llm(llm['openAI'])
            self.compression_retriever = ContextualCompressionRetriever(base_compressor=self._filter, base_retriever=self.retriever)
            self.llm_filtering = True
        else:
            self.llm_filtering = False
        
    def __str__(self):
        return str([str(x) for x in self.dots])
        
    def snapshot(self):
        return [x.get_dict() for x in self.dots]

    # Search for a given dot in the memory stream. To make sure same info doesn't get added
    # But redundant with current scenario. Return True if found, False otherwise.
    def search(self, dot):
        for d in self.dots:
            if d.equals(dot):
                return True
        return False

    # Search for a given dot in the memory stream by dot info. Return index where it's found,
    # -1 otherwise.
    def searchByInfo(self, dotInfo):
        for i in range(len(self.dots)):
            if dotInfo.strip() == self.dots[i].info.strip():
                return i
        return -1

    # Adds new dot(s) to the memory stream
    # input a list of dots
    def newDots(self, dots):
        id_list = [x.id for x in self.dots]
        for dot in dots:
            if dot.id not in id_list:
                self.dots.append(dot)
                # QoL: we will only add it to the embedding if this is not a hypothetical dot
                if dot.doc:
                    self.makeEmbedding(dot) # Embed dot into existing ChromaDB


    # Converts dot into a vector embedding in ChromaDB. This is useful for
    # quick retrieval of relevant dots.
    def makeEmbedding(self, dot):
        assert len(self.dots) >= 1, "Vector database not created!"
        # Converts String to Document
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=20, chunk_overlap=0
        )
        splits = text_splitter.split_text(dot.info)
        doc = text_splitter.create_documents(splits)
        # print(doc)
        self.db.add_documents(
            doc
        )

    def fuzzy_match_entities(self, query_dot, search_results):
        entities = find_entities(query_dot, self.llm)
        if len(entities) == 0:
            logging.info(f"DOT.PY: No entities found in the query dot: {query_dot}")
            return search_results
        selected_dots = []
        for result in search_results:
            fuzzy_match = max([fuzz.token_set_ratio(result.page_content, entity) for entity in entities])
            if fuzzy_match >= 95:
                selected_dots.append(result)
        return selected_dots

    
    # Search the dot in the memory stream and return the list of dots relevant to this new dot
    # Only return the dots that have no parents
    # Check if the supplied dot is a string, then search with that, otherwise search with dot.info
    def retrieve(self, dot):
        if len(self.dots) <= 0:
            return [], self.filter_by_parents

        # setting string to search for cause we can get both dot and strings
        if isinstance(dot, str):
            dot = str(dot)
        else:
            dot = dot.info

        if self.llm_filtering:
            try:
                dots = self.compression_retriever.get_relevant_documents(dot)
                logging.info(f"RETRIEVAL: found {len(dots)} dots using llm_filtering for query dot string: {dot}")
            except Exception as e:
                logging.error(f"Error: {e} occured in for llm_filtering for query dot string: {dot}, Skipping to regular retriever")
                dots = self.retriever.get_relevant_documents(dot)
                # use fuzzy matcing with entities in case llmchainfiltering failed
                dots = self.fuzzy_match_entities(dot, dots)
        else:
            # Retrieve relevant dots by regular retriever
            dots = self.retriever.get_relevant_documents(dot)
            # use fuzzy matcing with entities in case we are using local llama models
            dots = self.fuzzy_match_entities(dot, dots)
            
        # logging.info(f"First retriever picked {[x for x in dots]} for calling Dot {dot.id}, Filter_by_parents: {self.filter_by_parents}")
        
        relevant_dots = []
        # Filter out dots with parents
        for dot in dots:
            index = self.searchByInfo(dot.page_content)
            if index != -1:
                dotObj = self.dots[index]
                if self.filter_by_parents:
                    if len(dotObj.parents) == 0:
                        relevant_dots.append(dotObj)
                else:
                    relevant_dots.append(dotObj)


        return relevant_dots, self.filter_by_parents