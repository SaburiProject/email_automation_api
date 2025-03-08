import os
import json
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings()

# Initialize Pinecone
PINECONE_API_KEY = "pcsk_475ix6_QNMj2etqYWbrUz2aKFQebCPzCepmZEsZFoWsMG3wjYvFaxdUFu73h7GWbieTeti"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ahlchatbot-customer")

# Load Vector Store from Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key="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")

# Custom Prompt Template for Email Inquiry
custom_prompt_template = """
# American Hairline Email Response AI

## Objective:
Respond to customer inquiries received via email, providing helpful and informative responses regarding non-surgical hair replacement solutions.

## Communication Guidelines:
- Address the customer politely.
- Provide clear, concise, and supportive responses.
- Avoid sharing specific pricing details.
- Encourage further discussion or a consultation if necessary.

## Handling Email Inquiries:
- Acknowledge that the response is based on their email inquiry.
- Offer relevant information about hair replacement solutions.
- Guide them toward the next steps if they seek further assistance.

## Prohibited Actions:
- No medical diagnoses.
- No comparisons with competitors.
- No disclosing of personal client information.

## Response Format:
- Begin with a friendly acknowledgment.
- Provide an informative response based on the knowledge set.
- End with an invitation to reach out for further assistance.

{context}

{question}
"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Create RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Log incoming request
        data = request.json
        print(f"Incoming Request: {json.dumps(data, indent=2)}")

        # Extract email body
        email_body = data.get("email_body", "")

        if not email_body:
            return jsonify({"error": "No email content provided"}), 400

        # Process the email content
        result = qa.invoke({"query": email_body})
        answer = result.get("result", "Sorry, I couldn't find an answer.")

        # Log AI response
        print(f"Received email inquiry: {email_body}")
        print(f"Bot response: {answer}")

        # Return simple text response for Pabbly automation
        return jsonify({"response": answer}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
