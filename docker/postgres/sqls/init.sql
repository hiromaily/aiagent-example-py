CREATE EXTENSION vector;

CREATE TABLE items (
    id bigserial PRIMARY KEY, 
    embedding vector(3)
);

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536)
);
