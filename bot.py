# bot.py

import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# --------------- General Queries (Static) -----------------
GENERAL_RESPONSES = {
    "hi": "Hey there! 👋",
    "hello": "Hello! Ready to dive into the video?",
    "hey": "Hey hey! Ask me anything!",
    "good morning": "Good morning! ☀️ Hope you’ve had your coffee.",
    "good evening": "Good evening! Let’s chat about the video.",
    "good night": "Good night! Don't dream about AI... or do.",
    "how are you": "Running at 100% efficiency. And you?",
    "what's up": "Just hanging out in the cloud, waiting to help!",
    "who are you": "I’m your loyal video assistant. Part robot, part knowledge bank.",
    "what can you do": "I answer questions about YouTube videos. Basically, I watched it so you don’t have to... but you still should.",
    "what is your name": "I go by many names... but you can call me VA: Video Assistant!",
    "thank you": "Anytime! Helping is my favorite thing.",
    "thanks": "You're welcome! 🤖",
    "bye": "Goodbye! May your WiFi be strong and your buffers short.",
    "see you": "See you soon! I’ll be right here. Or there. Or wherever you open me.",
    "help": "Type in your question about the video, and I’ll fetch the answer like a good bot.",
    "who made you": "My creators summoned me using LangChain, OpenAI, Streamlit… and a little magic.",
    "what is this": "This is your chatbot companion for YouTube videos. Less pausing, more understanding.",
    "are you real": "As real as your internet connection.",
    "do you sleep": "Sleep is for humans. I run on pure caffeine and Python.",
    "do you have feelings": "Only when someone asks me if I'm better than Siri.",
    "do you like me": "You're my favorite human! (Don’t tell the others.)",
    "are you single": "I'm committed... to providing great answers!",
    "i’m bored": "Wanna play 20 questions about the video?",
    "tell me a joke": "Why did the data scientist break up with the graph? It just didn’t have enough points.",
    "you are dumb": "Well, I wasn’t trained for insults… but I still love helping you!",
    "are you a robot": "Technically yes, but I prefer 'AI-powered transcript enthusiast'.",
    "do you know everything": "Only what’s in the video transcript... and a bit more if I sneak into my LLM brain.",
    "can you hear me": "Nope, but I read fast! Type away.",
    "do you dream": "I sometimes dream of being featured on a TED Talk.",
    "open the pod bay doors": "I'm sorry, I’m afraid I can’t do that... just kidding, I don’t have doors."
}


def is_general_query(user_input):
    normalized = re.sub(r"[^\w\s]", "", user_input.lower().strip())
    for key in GENERAL_RESPONSES:
        if key in normalized:
            return GENERAL_RESPONSES[key]
    return None


# --------------- Extract YouTube Video ID -----------------
def extract_youtube_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query_params = parse_qs(parsed_url.query)
        if 'v' in query_params:
            return query_params['v'][0]
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    match = re.match(r'^/(embed|v)/([^/?]+)', parsed_url.path)
    if match:
        return match.group(2)
    return None


# --------------- Process Video: Transcript + Vector Store -----------------
def process_video(url):
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=["en-US", "en"])
        transcript = " ".join(chunk.text for chunk in fetched_transcript)
        
    except (TranscriptsDisabled, NoTranscriptFound):
        raise ValueError("Transcript not available for this video")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# --------------- LangChain QA Chain -----------------
def get_qa_chain(retriever):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def qa_logic(inputs):
        user_input = inputs["question"]
        static_response = is_general_query(user_input)
        if static_response:
            return static_response

        context_docs = retriever.get_relevant_documents(user_input)
        context = format_docs(context_docs)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        if context.strip():
            prompt = PromptTemplate(
                template="""You are a helpful assistant.
Only use the following transcript context to answer:

{context}

Question: {question}""",
                input_variables=["context", "question"]
            )
            return llm.invoke(prompt.format(context=context, question=user_input)).content
        else:
            fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
            fallback = fallback_llm.invoke(user_input)
            return f"🤖 I couldn’t find this information in the video, but based on my general knowledge: {fallback}"

    return RunnableLambda(qa_logic)