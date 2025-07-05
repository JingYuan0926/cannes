# AI Data Analyst Chatbot 👇
**Privacy-First, Enterprise-Grade AI Data Analysis with Secure Document Processing**

![Logo](https://github.com/JingYuan0926/cannes/blob/main/public/next.svg?raw=true)

A breakthrough AI-powered data analysis platform that processes PDF and DOCX documents within a secure environment, providing intelligent insights through conversational AI while maintaining complete data privacy and confidentiality.

For a comprehensive guide on setup and deployment, please refer to this [README Guide](https://github.com/JingYuan0926/cannes/blob/main/README.md).

We've deployed a live demonstration of **AI Data Analyst Chatbot** at [http://localhost:3000/chat](http://localhost:3000/chat).

---

## Inspiration: How We Came Up with This Idea 💡

In today's data-driven world, organizations and individuals constantly struggle with analyzing complex documents and extracting meaningful insights. Traditional data analysis tools often require technical expertise, while cloud-based AI services raise concerns about data privacy and security. Many users find themselves juggling between multiple tools—uploading documents to one service, copying results to another, and manually piecing together insights.

We thought:

> *"What if we could create a single, secure platform that combines document processing, AI analysis, and conversational interaction—all while keeping data completely private and accessible to anyone?"*

This exploration led us to design a privacy-first AI data analyst that operates within a secure environment, providing enterprise-grade analysis capabilities without compromising on security or ease of use.

---

## Important Terms ⚡

To better understand our AI Data Analyst Chatbot's documentation:

- **TEE (Trusted Execution Environment):** A secure computing environment that protects data and code from external access, ensuring privacy and confidentiality.
- **Session Management:** UUID-based tracking that maintains conversation context across multiple queries while keeping data isolated.
- **Document Sharding:** The process of breaking down large documents into manageable chunks for secure processing and analysis.

These terms help clarify how our platform securely processes documents and maintains user privacy throughout the analysis process.

---

## The Problem 🚧

Organizations and individuals face several significant challenges when attempting to analyze documents and extract insights:

**1. Privacy Concerns:** Sending sensitive documents to third-party AI services poses unacceptable risks for many organizations, especially when dealing with confidential business data or personal information.

**2. Technical Barriers:** Traditional data analysis tools require technical expertise, making them inaccessible to non-technical users who need insights from their documents.

**3. Fragmented Workflows:** Users often need to juggle multiple tools—document uploaders, analysis platforms, and reporting tools—creating inefficient and error-prone processes.

**4. Limited Context:** Most AI analysis tools provide one-off responses without maintaining conversation context, making it difficult to ask follow-up questions or explore deeper insights.

**5. File Format Limitations:** Many platforms only support specific file formats, forcing users to convert documents or lose valuable formatting and structure.

**6. Cost Prohibitive:** Enterprise-grade AI analysis tools often come with high costs, making them inaccessible to smaller organizations and individuals.

---

## The Solution 🔑

Our AI Data Analyst Chatbot addresses these challenges by implementing a secure, conversational document analysis platform:

**1. Privacy-First Architecture:** All document processing and AI analysis occurs within a secure environment, ensuring that sensitive data never leaves the user's control.

**2. Multi-Format Support:** Seamlessly processes PDF and DOCX documents while preserving formatting and structure for accurate analysis.

**3. Conversational Interface:** Maintains context throughout the session, allowing users to ask follow-up questions and explore insights naturally.

**4. Enterprise-Grade AI:** Powered by GPT-4, providing sophisticated analysis capabilities while maintaining the security and privacy of sensitive documents.

**5. Session Management:** UUID-based session tracking ensures data isolation and secure conversation continuity.

**6. Source Citations:** Includes reputable sources with clickable hyperlinks, providing transparency and credibility to AI-generated insights.

In short, our platform enables users to upload documents, ask questions in natural language, and receive intelligent, contextual responses—all while maintaining complete data privacy and security.

---

## How Our Project Works ⚙️

### Project Flow 1: Document Upload & Analysis

1. **Upload Documents**
   - Users upload PDF or DOCX files through our secure interface
   - Files are automatically parsed and processed within the secure environment
   - Document content is extracted and prepared for AI analysis

2. **Establish Context**
   - The AI system processes the uploaded documents to understand the content
   - A session ID is generated to maintain conversation context
   - Users receive confirmation that their documents are ready for analysis

3. **Conversational Analysis**
   - Users ask questions about their documents in natural language
   - The AI provides concise, actionable insights (2-4 sentences by default)
   - Follow-up questions maintain context from previous interactions

4. **Detailed Analysis (Optional)**
   - Users can request detailed explanations by explicitly asking for more information
   - The AI provides comprehensive analysis when requested
   - Source citations are included with clickable hyperlinks

### Project Flow 2: Direct Chat Analysis

1. **Start Conversation**
   - Users can begin chatting with the AI without uploading documents
   - The AI provides general data analysis insights and guidance
   - Context is maintained throughout the conversation

2. **Ask Questions**
   - Users pose questions about data analysis, trends, or insights
   - The AI responds with intelligent, well-formatted answers
   - Responses include relevant source citations when appropriate

3. **Follow-up Exploration**
   - Users can ask follow-up questions to dive deeper into topics
   - The AI maintains conversation context for coherent responses
   - Complex queries are broken down into manageable insights

---

## System Architecture High-Level Overview 🏗️

![Architecture](https://github.com/JingYuan0926/cannes/blob/main/public/next.svg?raw=true)

The current architecture consists of a Next.js frontend, OpenAI API integration, and secure document processing pipeline. This setup provides two clear user pathways:

### Document Analysis Flow (With File Upload - Blue Line)

- **File Upload**: Users upload PDF or DOCX files through the secure interface
- **Document Processing**: Files are parsed using pdf-parse and mammoth libraries
- **Session Management**: UUID-based session tracking maintains conversation context
- **AI Analysis**: GPT-4 processes document content and user queries
- **Response Generation**: Formatted responses with source citations are returned to users

### Direct Chat Flow (Without File Upload - Green Line)

- **User Input**: Users ask questions directly without uploading documents
- **Context Management**: Conversation history is maintained for coherent responses
- **AI Processing**: GPT-4 analyzes queries and provides intelligent insights
- **Response Delivery**: Formatted responses with relevant source citations

This dual-flow system architecture ensures flexibility, security, and user-friendly interaction while maintaining data privacy and providing enterprise-grade analysis capabilities.

---

## Document Processing Architecture 📦

![Document Processing](https://github.com/JingYuan0926/cannes/blob/main/public/next.svg?raw=true)

Our document processing pipeline handles multiple file formats securely:

- **PDF Processing**: Uses pdf-parse library to extract text while preserving structure
- **DOCX Processing**: Leverages mammoth library for Word document parsing
- **Security**: All processing occurs within secure environment with automatic file cleanup
- **Session Isolation**: Each user session is isolated with unique UUID tracking

The processed content is then made available to the AI for intelligent analysis and response generation.

---

## Tech Stack Overview 🛠️

- **Next.js 15.3.5** – Front-end React framework with API routes
- **OpenAI API** – GPT-4 integration for intelligent analysis
- **ReactMarkdown** – Rich text rendering with HTML support
- **Multer** – File upload handling and processing
- **PDF-Parse** – PDF document text extraction
- **Mammoth** – DOCX document processing
- **UUID** – Session management and tracking
- **Tailwind CSS** – UI styling and responsive design
- **Rehype-Raw** – HTML rendering for source citations
- **Remark Plugins** – Markdown formatting and processing

---

## Important Code Directories 📂

Here's a brief overview of important directories in our repository:

- **`/pages`**
  - **`chat.js`**: Main chat interface with file upload and conversation management
  - **`api/chatbot.js`**: AI API endpoint handling both file uploads and chat conversations
  - **`_app.js`**: App wrapper and global configuration

- **`/styles`**
  - **`globals.css`**: Global styles including ReactMarkdown formatting

- **`/public`**
  - Static assets and icons for the application

- **`/package.json`**
  - Project dependencies including OpenAI, document processing libraries, and UI components

---

## How We Are Different 🌟

Unlike traditional document analysis tools, our platform focuses on **conversational AI and privacy-first architecture** from the **user's perspective**. We're not just processing documents; we're creating an intelligent, secure, and accessible analysis experience. Here's how our **AI Data Analyst Chatbot** stands out:

| **Feature**                          | **Traditional Document Analysis**                                     | **AI Data Analyst Chatbot**                                                                 |
|--------------------------------------|-----------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Privacy & Security**               | Often requires uploading to third-party servers with limited privacy guarantees | **Privacy-first architecture** with secure document processing and automatic file cleanup |
| **Conversational Interface**         | Typically one-off analysis with no follow-up capability | **Maintains conversation context** allowing natural follow-up questions and deeper exploration |
| **Multi-Format Support**             | Often limited to specific file formats | **Seamless PDF and DOCX processing** with preserved formatting |
| **Source Citations**                 | Rarely includes source attribution | **Reputable sources with clickable hyperlinks** for transparency and credibility |
| **Response Formatting**              | Plain text or basic formatting | **Rich HTML formatting** with proper styling and structure |
| **Session Management**               | No conversation continuity | **UUID-based session tracking** for secure, isolated conversations |
| **Accessibility**                    | Often requires technical expertise | **User-friendly interface** accessible to non-technical users |
| **Cost Effectiveness**               | Expensive enterprise solutions | **Affordable AI analysis** powered by OpenAI's GPT-4 |

**AI Data Analyst Chatbot** goes beyond traditional document analysis by offering a **secure, conversational, and accessible solution** that prioritizes privacy while providing enterprise-grade analysis capabilities.

---

## Future Implementations 🚀

We are committed to continuously improving our platform and expanding its capabilities. Here are some exciting future plans:

### Enhanced Document Support
- **CSV Processing**: Add support for spreadsheet analysis and data visualization
- **Image Analysis**: Integrate OCR capabilities for image-based documents
- **Multi-Language Support**: Expand language processing capabilities

### Advanced AI Features
- **Custom Prompts**: Allow users to create and save custom analysis prompts
- **Batch Processing**: Enable analysis of multiple documents simultaneously
- **Export Capabilities**: Add functionality to export analysis results and insights

### Security Enhancements
- **End-to-End Encryption**: Implement additional encryption layers for enhanced security
- **Audit Logging**: Add comprehensive logging for compliance and transparency
- **Access Controls**: Implement role-based access controls for enterprise users

### Integration Capabilities
- **API Access**: Provide REST API for third-party integrations
- **Webhook Support**: Enable real-time notifications and integrations
- **Cloud Storage**: Add support for direct integration with cloud storage providers

---

## Team 👥

- **Phen Jing Yuan**  
  - *Role*: Project Lead, AI & Backend Integration
  - [LinkedIn](https://www.linkedin.com/in/jing-yuan-phen-b42266295/) | [Twitter](https://x.com/ilovedahmo)

---

## Getting Started 🚀

### Prerequisites
- Node.js 18+ 
- npm or yarn
- OpenAI API key

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/JingYuan0926/cannes.git
   cd cannes
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env.local
   # Add your OpenAI API key to .env.local
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the development server**
   ```bash
   npm run dev
   ```

The app will be accessible at [http://localhost:3000/chat](http://localhost:3000/chat).

---

## Usage Guide 📖

### **Getting Started**
1. Navigate to `http://localhost:3000/chat`
2. Upload PDF or DOCX files (optional)
3. Ask questions about your data
4. Receive AI-powered insights

### **File Upload**
- **Supported Formats**: PDF, DOCX
- **File Size Limit**: 5MB per file
- **Processing**: Files are automatically parsed and analyzed

### **Conversation Flow**
1. **Upload Phase**: Upload documents to establish context
2. **Analysis Phase**: Ask questions about your data
3. **Follow-up**: Continue the conversation with additional queries

### **Response Types**
- **Concise Mode**: Default short, actionable insights
- **Detailed Mode**: Comprehensive analysis when requested
- **Source Citations**: Includes reputable sources with links

---

## API Endpoints 🔧

### **POST /api/chatbot**
Handles both file uploads and chat conversations.

**File Upload:**
```javascript
const formData = new FormData();
formData.append('files', file);
formData.append('prompt', 'Analyze this document');

const response = await fetch('/api/chatbot', {
  method: 'POST',
  body: formData
});
```

**Chat Conversation:**
```javascript
const response = await fetch('/api/chatbot', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'What are the key trends?',
    sampleDataReady: true,
    conversation: [...],
    sessionId: 'uuid'
  })
});
```

---

## Security Features 🔒

- **File Validation**: Type and size checking
- **Temporary Storage**: Files deleted after processing
- **Session Isolation**: UUID-based session management
- **Error Handling**: Graceful error responses
- **Input Sanitization**: Safe handling of user inputs

---

## Contributing 🤝

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments 🙏

- **OpenAI** for providing the GPT-4 API
- **Next.js** for the excellent React framework
- **ReactMarkdown** for rich text rendering
- **PDF-Parse & Mammoth** for document processing

---

## Support 📞

For support, create an issue in this repository or contact the development team.

---

**Built with ❤️ using Next.js and OpenAI**
