# SEInsight👇
**An AI Agent Data Analyst build in TEE(Trusted Execution Environment)**
![MainPage](https://github.com/JingYuan0926/cannes/blob/main/public/MainPage.png?raw=true)


An AI-powered data analysis platform that processes data within a Trusted Execution Environment (TEE), providing intelligent insights through conversational AI while maintaining complete data privacy and confidentiality. No more data leaks, no more privacy concerns, just secure, powerful analysis that keeps your data safe.

We've deployed a live demonstration of **(SEInsight)** at [here](https://f81f-83-144-23-155.ngrok-free.app/).

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

**5. Enterprise-Grade AI:** Powered by OpenAI, providing sophisticated analysis capabilities while maintaining the security and privacy of sensitive documents.

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
   - The AI accesses the database inside the Trusted Execution Environment (TEE) and performs analysis using major machine learning libraries like **matplotlib, pandas, sklearn**, etc.
   - The AI automatically determines the most optimal analysis strategy based on the user’s data.
   - The flow of analysis will follow a structured process: **ETL (Extract, Transform, Load) → Preprocessing → Exploratory Data Analysis (EDA) → Machine Learning Analysis → Reporting**.
   - The AI generates insights based on this flow and provides concise, actionable findings (2-4 sentences by default).
   - Follow-up questions maintain context from previous interactions, ensuring continuous, relevant responses.
   - Users can view adaptive insight cards that display relevant information tailored to their needs.
   - A summary of all insight cards is provided for a comprehensive overview of the analysis.

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


The current architecture consists of a Next.js frontend, OpenAI API integration, and secure document processing pipeline. This setup provides two clear user pathways:

![MainPage](https://github.com/JingYuan0926/cannes/blob/main/public/Architecture.png?raw=true)


![UserFlow](https://github.com/JingYuan0926/cannes/blob/main/public/UserFlow.png?raw=true)



### Document Analysis Flow 

- **File Upload**: Users upload PDF or DOCX files through the secure interface
- **Document Processing**: Files are stored using Walrus
- **Session Management**: UUID-based session tracking maintains conversation context
- **AI Analysis**: GPT-4o Mini processes csv input and perform analysis using machine learning library tools
- **Response Generation**: Formatted responses with source citations and insight cards are returned to users

### Direct Chat Flow (Without File Upload - Green Line)

- **User Input**: Users ask questions directly without uploading encrypted data
- **Context Management**: Conversation history is maintained for coherent responses
- **AI Processing**: GPT-40 Mini analyzes queries and provides intelligent insights
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

## Sponsor Integration & Technical Contributions

### Oasis
Analysis ROFL: [rofl1qquxmr7y0ranr3man30020pwewdhan2udynt65k0](https://explorer.oasis.io/testnet/sapphire/rofl/app/rofl1qquxmr7y0ranr3man30020pwewdhan2udynt65k0?q=rofl1qquxmr7y0ranr3man30020pwewdhan2udynt65k0)

EDA ROFL: [rofl1qry08thamh4tedc36j5t3y3kawcll79dw5ysacfw](https://explorer.oasis.io/testnet/sapphire/rofl/app/rofl1qry08thamh4tedc36j5t3y3kawcll79dw5ysacfw?q=rofl1qry08thamh4tedc36j5t3y3kawcll79dw5ysacfw)

ETL ROFL: [rofl1qrczld5trq7psuan5p74uw60zh95rhawmg5df6ma](https://explorer.oasis.io/testnet/sapphire/rofl/app/rofl1qrczld5trq7psuan5p74uw60zh95rhawmg5df6ma?q=rofl1qrczld5trq7psuan5p74uw60zh95rhawmg5df6ma)

Preprocessing ROFL: [rofl1qp7mks4jqk74snwtnnwdhspcex4gl97s5gqw3nqc](https://explorer.oasis.io/testnet/sapphire/rofl/app/rofl1qp7mks4jqk74snwtnnwdhspcex4gl97s5gqw3nqc?q=rofl1qp7mks4jqk74snwtnnwdhspcex4gl97s5gqw3nqc)

### Walrus 
https://walruscan.com/testnet/blob/-GiYJQxw-QW2_XhGc0NFcB6OoKdxWUf8neI_fEVrn5U

### Artificial Superintelligence Alliance
https://agentverse.ai/agents/details/agent1qg869zt362lj4e60n5z6mvkmjmztnd2vu6k8ud3ee4fl4rrwea7ucqvhrqh/profile

---

## How We Are Different 🌟

Unlike traditional document analysis tools, our platform focuses on **conversational AI and privacy-first architecture** from the **user's perspective**. We're not just processing documents; we're creating an intelligent, secure, and accessible analysis experience. Here's how our **AI Agent Data Analyst** stands out:

| **Feature**                          | **Traditional  Analysis Bot**                                     | **AI Agent Data Analyst**                                                                 |
|--------------------------------------|-----------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Privacy & Security**               | Often requires uploading to third-party servers with limited privacy guarantees | **TEE-based architecture** with secure encrypted data processing and automatic cleanup |
| **Conversational Interface**         | Typically one-off analysis with no follow-up capability | **Maintains conversation context** allowing natural follow-up questions and deeper exploration |
| **Encrypted Data Processing**        | Often limited to specific file formats | **Seamless encrypted data processing** with complete security |
| **Source Citations**                 | Rarely includes source attribution | **Reputable sources with clickable hyperlinks** for transparency and credibility |
| **Response Formatting**              | Plain text or basic formatting | **Rich HTML formatting** with proper styling and structure |
| **Accessibility**                    | Often requires technical expertise | **User-friendly interface** accessible to non-technical users |
| **Cost Effectiveness**               | Expensive enterprise solutions | **Affordable AI analysis** powered by OpenAI's GPT-4 |

**AI Data Analyst Chatbot** goes beyond traditional document analysis by offering a **secure, conversational, and accessible solution** that prioritizes privacy while providing enterprise-grade analysis capabilities.

---

## Future Implementations 🚀

We are committed to continuously improving our platform and expanding its capabilities. Here are some exciting future plans:

### Stage 1
- **Better Machine Learning Model Performance**
- **Mass Support of Machine Learning Algorithm**
- **GPU TEE for Better LLM Performance**

- ### Before vs After Report System 🔁
- **Impact Tracking**: Monitor the effectiveness of AI recommendations over time
- **Visual Comparisons**: Generate before/after charts and metrics showing measurable improvements
- **ROI Measurement**: Demonstrate concrete value and return on investment from following AI advice
- **Decision Optimization**: Help users understand which recommendations yield the best results
- **Progress Visualization**: Show cumulative improvements and trends in decision-making effectiveness
  
### Stage 2
- **Access Control for Different Management Level**
- **Go Live on MainNet**

---

## Team 👥

- **Derek Liew Qi Jian**  
  - *Role*: Project Lead, AI & TEE Integration
  - [LinkedIn](https://www.linkedin.com/in/derek2403/) | [Twitter](https://x.com/derek2403)

- **Phen Jing Yuan**  
  - *Role*: Project Lead, AI & Backend Integration
  - [LinkedIn](https://www.linkedin.com/in/jing-yuan-phen-b42266295/) | [Twitter](https://x.com/ilovedahmo)

- **Yeoh Choon Xiang**  
  - *Role*: Frontend Developer & UI/UX Design 
  - [LinkedIn](https://www.linkedin.com/in/choon-xiang-yeoh/) | [Twitter](https://x.com/choonxiangg)

- **Choong Zhao Zheng**  
  - *Role*: Frontend Developer & UI/UX Design
  - [LinkedIn](https://www.linkedin.com/in/choong-zhao-zheng-business-information-systems/)

- **Yee Lee Ren (Johnny)**  
  - *Role*: Frontend Developer & UI/UX Design
  - [LinkedIn](https://www.linkedin.com/in/ee-mun-leong-700a23316/) | [Twitter](https://x.com/14jren)

