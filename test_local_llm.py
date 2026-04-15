from app.llm.local_llm_client import LocalLLMClient

client = LocalLLMClient()

response = client.generate("Explain what AI is in one sentence")
print(response)