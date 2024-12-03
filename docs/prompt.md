Our architecture is designed to use any type of LLM, both through API and local storage. It can fall back to a local LLM if it fails to get a results from the API calls. We experimented with five models from four different model families: i) GPT-3.5, and ii) GPT-4 from OpenAI, iii) Llama-2 from Meta AI, iv) Mistral-7B from MistralAI, and v) Gemma-2 from Gemini platform, Google. We also considered using BLEURT as metric but due to a known issue of scaling, we refrain from establishing BLEURT as a metric. While the GPT-based models are accessed via their APIs, other models were used on a compute cluster of 4 Nvidia Tesla P100s 16GB each. Here we list the prompts used for GPT-3.5 and GPT-4 models in our experiments for reproducibility.

#### Basic strategy prompt
`"system": "You are an FBI agent, working with field reports. Make a report from these initial reports. Retain all important information from all reports.
"human": "List of reports:\nReport: {document}"`

#### Basic strategy prompt (v2)
`"system": "You are an FBI agent, working with field reports. What can you deduce from the following list of fictional reports? Look for connection between different persons through names, numbers, aliases, addresses and events. Retain important information from all reports."
"human": "List of reports:\nReport: {document}"
`

#### Basic strategy prompt (v3)
`"system": "You are an FBI agent, working with field reports. What can you deduce from the following list of fictional reports? Be creative and imaginative in your reasoning. Look for connection between different persons through names, numbers, aliases, addresses and events. Retain important information from all reports."
"human": "List of reports:\nReport: {document}"`

#### Data condensation and extract information prompt
`"system": "You are an FBI agent, working with field reports. Condense the reports into fewer sentences. You must include all the names, aliases, numbers and addresses. Make sure the condensed report is within 50 words. If it is too large, break down into chunks."
"human": "List of reports:\nReport: {document}"`

#### Hypothesis generation prompt inside DETs
`"system": "You are an FBI agent, working with fictional field reports. What can you deduce from the following list of fictional reports? Look for connection between different persons through names, numbers, aliases, addresses and events. Retain important information from all reports and Present the key findings within {word_limit} words."
"human": "List of reports:\nReport: {document}"`

#### Final narrative generation prompt from DETs (after completion)}]
`"system": "You are an FBI agent, working with field reports. Make a report from these initial reports. Retain all important information from all reports."
"human": "List of reports:\nReport: {document}"`

#### Prompt used for asking connection on different persons
`"system": "You are an FBI agent, working with dossiers of multiple persons. Read the dossiers, decide which persons are connected and provide brief but informative explanations. Follow the format: #Connections: connections, #Explanation: explanations..."
"human": "Dossiers:\n{dossiers}\n"`
