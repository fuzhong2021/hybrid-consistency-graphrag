docker run \
    --name neo4j-graphrag \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/masterarbeit2024 \
    -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
    -v neo4j-data:/data \
    -v neo4j-logs:/logs \
    -d \
    neo4j:5.15.0