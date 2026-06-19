import os
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI","")
NEO4J_USER = os.getenv("NEO4J_USER","")
NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",""
)


class Neo4jConnection:
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            keep_alive=True,
            max_connection_lifetime=300,
            max_connection_pool_size=50,
            connection_timeout=15.0,
        )

    async def close(self):
        await self.driver.close()

    async def test_connection(self) -> bool:
        try:
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            return False

    async def fetch_user_graph_facts(self, user_id: str) -> str:
        """
        Fetches facts radiating from THIS user's specific User node only.

        SECURITY FIX: the original query matched ANY node labeled `User`
        with no filter, so all users' facts were mixed together. Now the
        User node is identified by its `user_id` property, and only facts
        connected to that exact node are returned.
        """
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[r]->(target)
            RETURN type(r) AS relation, target.name AS target_name, labels(target)[0] AS target_type
            """
            result = await session.run(query, user_id=user_id)
            records = await result.data()

            if not records:
                return "No permanent facts established yet."

            facts = []
            for row in records:
                facts.append(
                    f"User {row['relation']} {row['target_name']} (Type: {row['target_type']})"
                )

            return "\n".join(facts)

    async def execute_graph_update(self, graph_data: dict, user_id: str):
        """
        Writes extracted facts into the graph, scoped to this user.

        SECURITY FIX: previously, entity nodes were merged purely by `name`
        with no owner — so "Python" mentioned by User A and User B became
        the SAME node, and their facts/edges could bleed together. Now:
          1. Every non-User entity node also carries an `owner_id` property,
             and MERGE matches on (name + owner_id), so two users' nodes
             with the same name stay completely separate.
          2. The User node itself is identified by `user_id`, created if
             missing.
          3. Edges are matched within the same owner_id scope only.
        """
        async with self.driver.session() as session:
            # Ensure this user's own User node exists.
            await session.run(
                "MERGE (u:User {user_id: $user_id})",
                user_id=user_id,
            )

            # 1. Merge entity nodes, scoped to this user via owner_id.
            for node in graph_data.get("nodes", []):
                entity_type = node["entity_type"]
                # User nodes are special-cased: they ARE the user_id-keyed node.
                if entity_type == "User":
                    continue
                query = f"""
                MERGE (n:{entity_type} {{name: $name, owner_id: $owner_id}})
                """
                await session.run(query, name=node["entity_name"], owner_id=user_id)

            # 2. Merge edges, scoped to this user.
            if "edges" in graph_data:
                for edge in graph_data["edges"]:
                    source_id = edge.get("source") or edge.get("source_node")
                    target_id = edge.get("target") or edge.get("target_node")

                    if not source_id or not target_id:
                        continue

                    raw_relation = edge.get("relation", "RELATED_TO")
                    safe_relation = (
                        raw_relation.upper().replace(" ", "_").replace("-", "_")
                    )

                    # Match source/target within this user's scope only.
                    # The User node matches on user_id; other entities match
                    # on name + owner_id (set in step 1).
                    query = f"""
                    MATCH (source) WHERE
                        (source:User AND source.user_id = $owner_id)
                        OR (source.name = $source_id AND source.owner_id = $owner_id)
                    MATCH (target) WHERE
                        (target:User AND target.user_id = $owner_id)
                        OR (target.name = $target_id AND target.owner_id = $owner_id)
                    MERGE (source)-[r:{safe_relation}]->(target)
                    """
                    await session.run(
                        query,
                        source_id=source_id,
                        target_id=target_id,
                        owner_id=user_id,
                    )


# Instantiate a singleton client for FastAPI to use
neo4j_client = Neo4jConnection()
