"""Custom Agent Use Case."""

#from infrastructure.openai_model import call_openai_api

class CustomTechnicalAgent:
    def __init__(self, openai_client):
        self.openai_client = openai_client

    def query(self, user_query):
        """Custom agent function that uses OpenAI API and LlamaIndex"""
        # Initial prompt
        instructions = "あなたは経験豊富なソフトウェアエンジニアです。"
        prompt = f"""
        ユーザーの知りたい技術に答えるために：
        1. その技術の概要について答えてください
        2. その技術を習得するために、Step by stepのガイダンスを提供してください
        
        ユーザーの知りたい技術: {user_query}
        """

        # Step 1: Initial thought generation using OpenAI API
        initial_thought = self.openai_client.call_openai_api(instructions, prompt)
        return initial_thought
