import os
import json
import asyncio
import networkx as nx
from networkx.readwrite import json_graph

class GraphMemoryService:
    """
    Manages long-term relational memory using a networkx MultiDiGraph.
    Stores facts as triples (Subject, Relation, Object) and persists them to a JSON file.
    """
    def __init__(self, file_path: str = None):
        if file_path is None:
            # Place storage inside backend/storage/graph_memory.json
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_dir = os.path.join(base_dir, "storage")
            self.file_path = os.path.join(storage_dir, "graph_memory.json")
        else:
            self.file_path = file_path
        
        self.lock = asyncio.Lock()
        self.graph = nx.MultiDiGraph()
        self.load_graph()

    def load_graph(self):
        """Loads the graph from the JSON file if it exists, otherwise initializes an empty MultiDiGraph."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = json_graph.node_link_graph(data)
                print(f"[GraphMemory] Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            except Exception as e:
                print(f"[GraphMemory] Failed to load graph from {self.file_path}: {e}. Starting with an empty graph.")
                self.graph = nx.MultiDiGraph()
        else:
            self.graph = nx.MultiDiGraph()

    def save_graph(self):
        """Persists the graph to the JSON file, creating parent directories if needed."""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            data = json_graph.node_link_data(self.graph)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[GraphMemory] Saved graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges to {self.file_path}.")
        except Exception as e:
            print(f"[GraphMemory] Failed to save graph to {self.file_path}: {e}")

    async def add_triple(self, subject: str, relation: str, object_: str):
        """
        Adds a relational triple (subject, relation, object_) to the graph and persists it.
        """
        async with self.lock:
            sub = subject.strip()
            rel = relation.strip()
            obj = object_.strip()
            
            if not sub or not rel or not obj:
                return

            # Check if this exact edge already exists to prevent duplicate parallel edges
            exists = False
            if self.graph.has_edge(sub, obj):
                for edge_key in self.graph[sub][obj]:
                    if self.graph[sub][obj][edge_key].get("relation") == rel:
                        exists = True
                        break
            
            if not exists:
                self.graph.add_edge(sub, obj, relation=rel)
                self.save_graph()

    async def get_triples(self) -> list[tuple[str, str, str]]:
        """Returns all triples currently in the graph."""
        async with self.lock:
            triples = []
            for u, v, key, data in self.graph.edges(keys=True, data=True):
                triples.append((u, data.get("relation", ""), v))
            return triples

    async def search_adjacent_nodes(self, query: str) -> list[tuple[str, str, str]]:
        """
        Tokenizes/normalizes query and searches for nodes containing any of the query terms.
        For matching nodes, returns adjacent triples (both incoming and outgoing).
        """
        async with self.lock:
            if not query:
                return []
            
            # Simple normalization of query
            query_normalized = query.lower()
            # Tokenize query into words, removing punctuation and short/filler words
            words = [w.strip("?,.!:;()\"'") for w in query_normalized.split()]
            # Filter terms to search (keep words of length > 2)
            query_terms = [w for w in words if len(w) > 2]
            
            if not query_terms:
                return []

            matched_nodes = set()
            for node in self.graph.nodes:
                node_lower = str(node).lower()
                # Check if any query term is a substring of the node name
                if any(term in node_lower for term in query_terms):
                    matched_nodes.add(node)

            triples = set()
            for node in matched_nodes:
                # Add all outgoing edges
                for neighbor in self.graph.successors(node):
                    for edge_key in self.graph[node][neighbor]:
                        rel = self.graph[node][neighbor][edge_key].get("relation", "")
                        triples.add((node, rel, neighbor))
                
                # Add all incoming edges
                for parent in self.graph.predecessors(node):
                    for edge_key in self.graph[parent][node]:
                        rel = self.graph[parent][node][edge_key].get("relation", "")
                        triples.add((parent, rel, node))

            return list(triples)
