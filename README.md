# 🤖 Chit-Chat With YouTube

Have you ever wished you could **chat with a YouTube video** like it’s a podcast guest? This app lets you do exactly that.

Just paste a YouTube URL, and the app extracts the transcript, embeds it, and lets you **ask questions** in a conversational interface powered by OpenAI's ChatGPT.


## 📌 What This Project Does

- Takes a **YouTube video link**
- Extracts its **transcript** using `youtube-transcript-api`
- Splits and embeds the transcript using **LangChain** + **FAISS**
- Allows the user to **ask questions** about the video
- Uses **OpenAI GPT model** to respond based on the transcript
- Handles **general greetings** and **fallback answers** using LLM when transcript lacks information


## 🧠 How It Works

1. **User pastes a YouTube link**  
   → The app extracts the video ID and fetches the transcript (if available).

2. **Transcript processing**  
   → The full transcript is chunked into small pieces and embedded using OpenAI embeddings.

3. **Embedding store (vector DB)**  
   → FAISS is used to store and retrieve relevant chunks based on user queries.

4. **Chat interface with fallback**  
   → If the answer is found in the transcript, it’s used.  
   → If not, the app replies with:  
   > _"I could not find this information in the video, but according to my knowledge..."_

5. **General queries** like "hi", "how are you?" are also handled separately to create a better experience.


## 🔧 Tech Stack

| Component           | Tool/Library                    |
|--------------------|----------------------------------|
| Frontend           | Streamlit                        |
| Transcript API     | `youtube-transcript-api`         |
| LLM                | OpenAI GPT-3.5/4                 |
| Vector DB          | FAISS                            |
| Embeddings         | OpenAI Embeddings via LangChain  |
| Retrieval & Prompt | LangChain                        |
| Programming Lang   | Python                           |


## 🛠️ Project Structure
Chit_Chat_With_Youtube/
├── app.py # Streamlit UI
├── bot.py # LangChain logic (transcript fetch, embedding, chat)
├── requirements.txt # Python dependencies
├── .env # Contains OPENAI_API_KEY (not pushed to GitHub)
├── README.md # You're reading it

## 📸 Final Output 
<img width="1896" height="962" alt="image" src="https://github.com/user-attachments/assets/680e794d-67f0-4b6f-a126-6347c176dd89" />


