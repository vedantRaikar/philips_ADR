import requests
import networkx as nx
import matplotlib.pyplot as plt
from SPARQLWrapper import SPARQLWrapper, JSON

def get_wikidata_id(drug_name):
    """Retrieve the Wikidata ID for a given drug name."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": drug_name,
        "language": "en",
        "type": "item"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data["search"]:
        return data["search"][0]["id"]
    else:
        raise ValueError(f"No Wikidata ID found for drug: {drug_name}")

def get_drug_interactions(drug_wikidata_id, drug_name, limit=20):
    """Query Wikidata for drug-drug, drug-food, and other interactions of a specific drug."""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>

    SELECT ?interactionEntityLabel ?interactionTypeLabel
    WHERE {{
      {{
        # Drug-drug interactions
        wd:{drug_wikidata_id} wdt:P129 ?interactionEntity .
      }}
      UNION
      {{
        # Drug-food interactions (e.g., food products)
        wd:{drug_wikidata_id} wdt:P1019 ?interactionEntity .
      }}
      UNION
      {{
        # Interactions with substances (e.g., alcohol, chemicals)
        wd:{drug_wikidata_id} wdt:P769 ?interactionEntity .
      }}
      UNION
      {{
        # Interactions with biological processes or effects
        wd:{drug_wikidata_id} wdt:P2175 ?interactionEntity .
      }}
      OPTIONAL {{ wd:{drug_wikidata_id} wdt:P2175 ?interactionType }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Add a User-Agent header
    sparql.addCustomHttpHeader("User-Agent", "MyApp/1.0 (https://example.com)")

    try:
        print("SPARQL Query:", query)  # Debugging: Log the query
        results = sparql.query().convert()
        if not results["results"]["bindings"]:
            raise ValueError(f"No interactions found for drug: {drug_name}")
        interactions = []
        for result in results["results"]["bindings"]:
            interaction_entity = result["interactionEntityLabel"]["value"]
            interaction = result.get("interactionTypeLabel", {}).get("value", "Unknown")
            interactions.append((drug_name, interaction_entity, interaction))
        return interactions
    except Exception as e:
        raise ValueError(f"Error fetching interactions for {drug_name}: {e}")

def visualize_graph(interactions, drug_name):
    """Visualize drug interactions using NetworkX and Matplotlib and return the plot."""
    G = nx.DiGraph()
    for drug1, drug2, interaction in interactions:
        G.add_node(drug1, color='red')
        G.add_node(drug2, color='blue')
        G.add_edge(drug1, drug2, label=interaction)
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    plt.figure(figsize=(10, 6))
    nx.draw(
        G, pos, with_labels=True, node_color=node_colors, edge_color="gray",
        font_size=10, node_size=2000, font_weight="bold"
    )
    edge_labels = {(drug1, drug2): interaction for drug1, drug2, interaction in interactions}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)
    plt.title(f"Drug Interactions for {drug_name}")
    return plt