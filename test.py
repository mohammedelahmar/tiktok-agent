import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_node("Employé", shape="box", label="Employé\n-\nnumSecu: String\netatCivil: String\nadresse: String\nemploi: String\nsalaire: double\n+\nafficherSalaire(): double\ngetSuperieur(): Employé\ngetSubordonnes(): List<Employé>")
G.add_node("Patron", shape="box", label="Patron\n-\nprimeRisque: double\n+\nafficherSalaire(): double")
G.add_node("Vendeur", shape="box", label="Vendeur\n-\ncommission: double\n+\nafficherSalaire(): double")
G.add_node("Caissière", shape="box", label="Caissière\n+\nafficherSalaire(): double")
G.add_node("Entreprise", shape="box", label="Entreprise\n-\nemployes: List<Employé>\n+\nsalaireTotal(): double\nafficherOrganigramme()")

# Add edges (inheritance)
G.add_edge("Employé", "Patron")
G.add_edge("Employé", "Vendeur")
G.add_edge("Employé", "Caissière")

# Draw the graph
plt.figure(figsize=(12, 10))
pos = graphviz_layout(G, prog="dot")
nx.draw(G, pos, with_labels=True, node_size=5000, node_color="lightblue", 
        font_size=8, arrows=True)
plt.savefig("uml_employes.png")
plt.show()