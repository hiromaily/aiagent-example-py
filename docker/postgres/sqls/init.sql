CREATE EXTENSION vector;

-- Vector dimensionality: 768
CREATE TABLE embeddings (
    id bigserial PRIMARY KEY, 
    embedding vector(768)
);

CREATE TABLE item_contents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(768)
);

-- Vector dimensionality: 1536
CREATE TABLE embeddings_large (
    id bigserial PRIMARY KEY, 
    embedding vector(1536)
);

CREATE TABLE item_contents_large (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536)
);
