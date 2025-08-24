# AI Chatbot with LangChain and Groq

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green.svg)

A sophisticated AI chatbot application built with Streamlit, LangChain, and Groq API. This project demonstrates how to create a modern conversational AI interface with advanced language models.

![Chatbot Demo](https://user-images.githubusercontent.com/placeholder/demo.gif)

## 🌟 Features

- **Intuitive Chat Interface**: Clean and responsive UI for natural conversations
- **Multiple LLM Support**: Choose from various Groq AI models like Llama 3, Mixtral, and Gemma
- **Customizable Parameters**: Adjust temperature and token limits for tailored responses
- **Advanced Prompt Engineering**: Customize system prompts to control AI behavior
- **Error Handling**: Robust error management for API issues and rate limits
- **Chat History**: Persistent chat sessions within the application
- **Responsive Design**: Works well on both desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- A Groq API key (get one from [Groq Console](https://console.groq.com))

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/SurajKhodade15/chatbot_langchain_genai.git
   cd chatbot_langchain_genai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables (optional):
   Create a `.env` file in the project root with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional for tracking
   LANGCHAIN_PROJECT=your_langchain_project_name  # Optional for tracking
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and go to `http://localhost:8501`

## 💡 Usage

1. Enter your Groq API key in the sidebar (if not set via environment variables)
2. Select your preferred AI model
3. Adjust advanced settings as needed (temperature, token limit, system prompt)
4. Type your message in the chat input and press Enter
5. View the AI's response and continue the conversation

## 🧠 Technical Implementation

This project demonstrates several key concepts and technologies:

### LangChain Integration
- **Chain Construction**: Uses LangChain's composable chain architecture
- **Prompt Templates**: Implements structured prompts with system and user messages
- **Output Parsing**: Handles model output with appropriate parsers

### Streamlit UI/UX
- **Custom CSS**: Enhanced UI with tailored styling
- **Session State**: Manages chat history and application state
- **Responsive Layout**: Adapts to different screen sizes

### Error Handling
- **API Error Management**: Handles authentication, rate limiting, and server errors
- **User Feedback**: Provides clear error messages and guidance
- **Graceful Degradation**: Maintains functionality when services are unavailable

### Code Architecture
- **Modular Design**: Separates concerns for improved maintainability
- **Type Hints**: Uses Python type annotations for better code quality
- **Clean Code Practices**: Follows PEP 8 style guide

## 🔧 Advanced Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key for authentication
- `LANGCHAIN_API_KEY`: Optional LangChain API key for tracking
- `LANGCHAIN_PROJECT`: Optional project name for LangChain tracking

### System Prompts

The system prompt controls how the AI assistant behaves. The default prompt is:

```
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
If you don't know the answer to a question, don't share false information.
```

You can customize this prompt in the Advanced Settings section to specialize the assistant for different domains or personalities.

## 🛠️ Project Structure

```
chatbot_langchain_genai/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🔄 Potential Enhancements

Future improvements could include:

- Document upload and analysis capabilities
- Integration with knowledge bases and databases
- Voice input/output support
- Multi-modal capabilities (images, audio)
- Persistent conversation storage
- User authentication system
- Additional model providers beyond Groq

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Groq API Documentation](https://console.groq.com/docs)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created by Suraj Khodade

Feel free to reach out or contribute to this project!

---

**Note**: This project is for educational purposes only. Be sure to comply with the terms of service for all APIs used.
