import main as main

import networkx as nx
from pyvis import network as net
import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import webbrowser


def create_nodes(g, nx_graph):
    """
    Customize nodes in the graph to each represent a dot.
    The node will contain information about the dot, including:
        - ID
        - Dcoument source (if applicable)
        - Info (document content or hypothesis)
        - Color (black or hypothesis)
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
    for node in nx_graph.nodes():
        doc_id = eval(node).get('id')
        info = eval(node).get('info')
        doc_name = eval(node).get('doc')
        doc_children = eval(node).get('children_dots')

        # Construct node label with doc_name and information about children dots
        

        # Set node color based on conditions
        if doc_name:
            color = 'black'  # If it has a doc_name but no children dots
            label_lines = [info[i:i+30] for i in range(0, len(info), 30)]
            label = '\n'.join(label_lines)
            label = f"ID:{doc_id}, DOC: {doc_name}\n"+label
            g.add_node(doc_id, label=label, children=doc_children, font='5px', size=4, color=color)
        else:
            label_lines = text_splitter.split_text(info)
            label = '\n'.join(label_lines)
            label = f"ID:{doc_id} \n"+label
            g.add_node(doc_id, label=label, font='5px', size=4)


def create_edges(g, nx_graph):
    """
    Add edges to the graph, where applicable, to represent connections between 
    two dots.
    """
    for edge in nx_graph.edges():
        source_id = eval(edge[0]).get('id')
        target_id = eval(edge[1]).get('id')

        g.add_edge(source_id, target_id)
        # g.add_edge(source_id, target_id, arrow=True, physics=True, smooth=dict(enabled=True, type='curvedCW')


def create_sidebar(file_name):
    """
    Post-processing script that will create a panel/sidebar for displaying dot 
    variables information. When a node (dot) is clicked, the sidebar will display the 
    dot's information in a clean, readable format.

    Black dot example:
        Black Dot
        ID: 121
        DOC: fbi10
        Steven Clark, employed by the City Computer Services Corp., failed to report his 
        arrest and conviction for assault and battery on his application for a 
        NYSE vendor's ID.

    Hypothesis dot example:
        Hypothesis Dot
        ID: 157
        The evidence indicates the existence of a criminal network involved in passport 
        forgery and illegal goods transportation, potentially with extremist ties. There 
        may be a connection between this network and the Seattle area, where intercepted 
        packages marked as "Home Made Candies" are suspected to be package bombs. Additionally, 
        the arrest of two American Airlines baggage handlers in Chicago suggests potential 
        involvement of airport staff in illegal activities.
    """
    with open(file_name, 'a') as f:
        f.write('''
        <!-- Add a sidebar for displaying node data -->
        <div id="sidebar" style="display: none; position: fixed; right: 0; top: 0; width: 30%; height: 100%; background-color: #f0f0f0; padding: 20px;">
            <h2 id="dot-title"></h2>
            <h4 id="dot-id"></h4>
            <h4 id="dot-docs"></h4>
            <div id="dot-info"></div>
            <div id="dot-children"></div>
        </div>

        <script>
            network.on("click", function(params) {
                if (params.nodes.length > 0) {
                    // A node was clicked
                    var nodeId = params.nodes[0];
                    var nodeData = nodes.get(nodeId);

                    var id = nodeData.id;
                    document.getElementById("dot-id").innerText = "ID: " + id;

                    if (nodeData.color !== "black") {
                        document.getElementById("dot-title").innerText = "Hypothesis Dot";
                        var label = nodeData.label.replace(/\\n/g, "");
                        document.getElementById("dot-docs").innerText = "";
                        document.getElementById("dot-info").innerText = label;
                        document.getElementById("dot-children").innerText = children;
                    } 
                    else {
                        // Split the label by the first newline character, and then
                        var labelSplit = nodeData.label.replace(/\\n/g, "");
                        if (labelSplit.includes("ID:")) { // ID is included in the label
                            var docs = labelSplit.split(".txt")[0].split(", ")[1];
                        }
                        else {
                            var docs = labelSplit.split(".txt")[0];
                        }
                        var info = labelSplit.split(".txt")[1];
                        var children = nodeData.children
                        document.getElementById("dot-docs").innerText = docs;
                        document.getElementById("dot-info").innerText = info;
                        document.getElementById("dot-title").innerText = "Black Dot";
                        document.getElementById("dot-children").innerText = children;
                        document.getElementById("dot-children").innerText = "";
                    }

                    document.getElementById("sidebar").style.display = "block";
                } else {
                    // No node was clicked, hide the sidebar
                    document.getElementById("sidebar").style.display = "none";
                }
            });
        </script>
        ''')


def adjust_height(file_name):
    """
    Post-processing script that will adjust the height of the network graph and loading bar
    in the HTML file, increasing the visibility window.
    """
    with open(file_name, 'r+') as f:
        content = f.read()

        # Use regex to replace the height of #mynetwork and #loadingBar
        content = re.sub(r'(#mynetwork\s*\{\s*[^}]*?height:\s*)\d+px', r'\g<1>900px', content)
        content = re.sub(r'(#loadingBar\s*\{\s*[^}]*?height:\s*)\d+px', r'\g<1>900px', content)

        # Write the modified content back to the file
        f.seek(0)
        f.write(content)
        f.truncate()


def postprocess_visuals(file_name):
    """
    This will run the necessary processing scripts enhance the visual representation 
    of the graph.
    """
    create_sidebar(file_name)
    adjust_height(file_name)


def create_UI(graph, file_name, load_memory_stream):
    """
    Develop a visual representation of the graph created by the pipeline.
    The results will be saved in an HTML file, and a post-processing script will
    be run to enhance the visual representation of the graph. The graph will 
    automatically open in the browser once the script is complete.
    """
    # 'Hierarchial' layout creates tree-like structure
    g = net.Network(directed =True)
    g.show_buttons(filter_=['layout'])

    #nx_graph = main.create_graph(main.save_load_memory_stream('type-v2_dataset-crescent_relevance-False_parent-False_additional-False_doc-True.pickle', mode='load'))
    if load_memory_stream:
        nx_graph = main.create_graph(main.save_load_memory_stream(graph, mode='load'))
    else:
        nx_graph = graph

    print([node.id for node in nx_graph.nodes()])
    print([(edge[0].id, edge[1].id) for edge in nx_graph.edges()])

    # Convert node IDs to strings if they are not already
    for node in nx_graph.nodes():
        if not isinstance(node, (str, int)):
            nx_graph = nx.relabel_nodes(nx_graph, {node: str(node)})

    create_nodes(g, nx_graph) # Add nodes (dots) to the graph
    create_edges(g, nx_graph) # Add edges (connections / parents) to the graph

    g.show(file_name, notebook=False)   # Create HTML file

    postprocess_visuals(file_name) # Enhance the visual representation of the graph

    webbrowser.open(file_name) # Display the HTML file in the browser

