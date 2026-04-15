from app.llm.llm_provider import LLMProvider

provider = LLMProvider()

print("Testing LLM switch...")
response = provider.generate_simple_response("Explain AI in one sentence")
print("Response:", response)