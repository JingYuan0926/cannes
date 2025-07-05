# AI Data Analyst (Name) 👇
**An AI Agent Data Analyst build in TEE(Trusted Execution Environment)**

An AI-powered data analysis platform that processes data within a Trusted Execution Environment (TEE), providing intelligent insights through conversational AI while maintaining complete data privacy and confidentiality. No more data leaks, no more privacy concerns — just secure, powerful analysis that keeps your data safe.

For a comprehensive guide on setup and deployment, please refer to this [README Guide](https://github.com/JingYuan0926/cannes/blob/main/README.md).

We've deployed a live demonstration of **(NAME)** at [http://localhost:3000/chat](http://localhost:3000/chat).

---

## Inspiration: How We Came Up with This Idea 💡

We noticed two critical problems in the data analysis landscape:

**For Small and Medium Enterprises (SMEs):**
Hiring a data analyst is prohibitively expensive. The average salary for a data analyst ranges from $60,000 to $120,000 annually, plus benefits and training costs. For SMEs operating on tight budgets, this is simply unaffordable, leaving them without access to crucial business insights that could drive growth and efficiency.

**For Fortune 500 Enterprises and Governments:**
Data leakage is a constant threat. High-profile breaches—from government agencies to Fortune 500 companies—make headlines weekly. Even with expensive security measures, sensitive data often ends up in the wrong hands, causing financial losses, reputational damage, and regulatory penalties.

We thought:

> *"What if we could create an AI Agent data analyst that's both affordable for SMEs and secure enough for enterprises and governments?"*

This exploration led us to design a privacy-first AI data analyst that operates within a Trusted Execution Environment (TEE), providing enterprise-grade analysis capabilities at a fraction of the cost while ensuring complete data security and confidentiality.

---

## The Problem 🚧

Organizations and individuals face several significant challenges when attempting to analyze documents and extract insights:

**1. Privacy Concerns:** Sending sensitive documents to third-party AI services poses unacceptable risks for many organizations, especially when dealing with confidential business data or personal information.

**2. Technical Barriers:** Traditional data analysis tools require technical expertise, making them inaccessible to non-technical users who need insights from their documents.

**3. Fragmented Workflows:** Users often need to juggle multiple tools—document uploaders, analysis platforms, and reporting tools—creating inefficient and error-prone processes.

**4. Limited Context:** Most AI analysis tools provide one-off responses without maintaining conversation context, making it difficult to ask follow-up questions or explore deeper insights.

**5. Cost Prohibitive:** Enterprise-grade AI analysis tools often come with high costs, making them inaccessible to smaller organizations and individuals.

---

## The Solution 🔑

Our AI Data Analyst Chatbot addresses these challenges by implementing a secure, conversational document analysis platform:

**1. Privacy-First Architecture:** All document processing and AI analysis occurs within Trusted Execution Environments (TEEs), ensuring that sensitive data aren't accessible to anyone, not even users themselves.

**2. Adaptive Insight Cards:** Dynamically displays relevant insight cards based on user needs and data content, hiding irrelevant information to provide focused, actionable intelligence. Users can view both individual insight cards and a comprehensive summary of all insights.

**3. Three-Tier Analysis:** Provides descriptive (what happened), predictive (what will happen), and prescriptive (what should be done) analysis capabilities for comprehensive data insights.

**4. Conversational Interface:** Maintains context throughout the session, allowing users to ask follow-up questions and explore insights naturally.

**5. Enterprise-Grade AI:** Powered by GPT-4, providing sophisticated analysis capabilities while maintaining the security and privacy of sensitive documents.

**6. Source Citations:** Includes reputable sources with clickable hyperlinks, providing transparency and credibility to AI-generated insights.

In short, our platform enables users to upload encrypted data, ask questions in natural language and receive intelligent, contextual responses—all while maintaining complete data privacy and security.

---

## How Our Project Works ⚙️

### Project Flow 1: Document Upload & Analysis

1. **Upload Encrypted Data**
   - Users upload encrypted data through our secure interface
   - Data is automatically processed within the secure environment
   - Encrypted content is decrypted and prepared for AI analysis

2. **Establish Context**
   - The AI system processes the uploaded encrypted data to understand the content
   - A session ID is generated to maintain conversation context
   - Users receive confirmation that their data is ready for analysis

3. **Conversational Analysis**
   - Users ask questions about the data in natural language
   - The AI provides concise, actionable insights (2-4 sentences by default)
   - Follow-up questions maintain context from previous interactions
   - Users can view adaptive insight cards that display relevant information based on their needs
   - A summary of all insight cards is provided for comprehensive overview

4. **Three Types of Analysis**
   - **Descriptive Analysis**: Understand what happened in the past with detailed data summaries and trends
   - **Predictive Analysis**: Forecast future outcomes and trends based on historical patterns
   - **Prescriptive Analysis**: Provide actionable recommendations and next steps for optimal decision-making

5. **Detailed Analysis (Optional)**
   - Users can request detailed explanations by explicitly asking for more information
   - The AI provides comprehensive analysis when requested
   - Source citations are included with clickable hyperlinks

---

## System Architecture High-Level Overview 🏗️

![Architecture](https://github.com/JingYuan0926/cannes/blob/main/public/next.svg?raw=true)

The current architecture consists of a Next.js frontend, OpenAI API integration, and secure document processing pipeline. This setup provides two clear user pathways:

### Document Analysis Flow (With File Upload - Blue Line)

- **File Upload**: Users upload PDF or DOCX files through the secure interface
- **Document Processing**: Files are parsed using pdf-parse and mammoth libraries
- **Session Management**: UUID-based session tracking maintains conversation context
- **AI Analysis**: GPT-4 processes document content and user queries
- **Response Generation**: Formatted responses with source citations and insight cards are returned to users

### Direct Chat Flow (Without File Upload - Green Line)

- **User Input**: Users ask questions directly without uploading encrypted data
- **Context Management**: Conversation history is maintained for coherent responses
- **AI Processing**: GPT-4 analyzes queries and provides intelligent insights
- **Response Delivery**: Formatted responses with relevant source citations and insight cards

This dual-flow system architecture ensures flexibility, security, and user-friendly interaction while maintaining data privacy and providing enterprise-grade analysis capabilities.

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
| **Privacy & Security**               | Often requires uploading to third-party servers with limited privacy guarantees | **Privacy-first architecture** with secure encrypted data processing and automatic cleanup |
| **Conversational Interface**         | Typically one-off analysis with no follow-up capability | **Maintains conversation context** allowing natural follow-up questions and deeper exploration |
| **Encrypted Data Processing**        | Often limited to specific file formats | **Seamless encrypted data processing** with complete security |
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
1. **Upload Phase**: Upload encrypted data to establish context
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
