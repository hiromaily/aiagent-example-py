CREATE EXTENSION vector;

CREATE TABLE embeddings (
    id bigserial PRIMARY KEY, 
    embedding vector(1536)
);

CREATE TABLE item_contents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536)
);
