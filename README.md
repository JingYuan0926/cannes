# SEInsight👇
**An AI Agent Data Analyst build in TEE(Trusted Execution Environment)**
![MainPage](https://github.com/JingYuan0926/cannes/blob/main/public/MainPage.svg?raw=true)


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

![Architecture](https://github.com/JingYuan0926/cannes/blob/main/public/Architecture.svg?raw=true)

The current architecture consists of a Next.js frontend, OpenAI API integration, and secure document processing pipeline. This setup provides two clear user pathways:

![Architecture](https://github.com/JingYuan0926/cannes/blob/main/public/Architecture.svg?raw=true)

![UserFlow](https://github.com/JingYuan0926/cannes/blob/main/public/UserFlow.svg?raw=true)



### Document Analysis Flow 

- **File Upload**: Users upload PDF or DOCX files through the secure interface
- **Document Processing**: Files are stored using Walrus
- **Session Management**: UUID-based session tracking maintains conversation context
- **AI Analysis**: GPT-4o Mini processes document content and user queries
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












# 🚀 AI Data Analysis Pipeline

A comprehensive AI-powered data analysis system with microservices architecture, featuring ETL processing, data preprocessing, exploratory data analysis (EDA), and machine learning analysis with TEE (Trusted Execution Environment) attestation.

## 🏗️ Architecture

The pipeline consists of 4 microservices:

- **ETL Service** (Port 3030) - Data extraction, transformation, and loading
- **Preprocessing Service** (Port 3031) - Data cleaning and preparation
- **EDA Service** (Port 3035) - Exploratory data analysis and visualization
- **Analysis Service** (Port 3040) - Machine learning analysis with TEE attestation

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed and running
- OpenAI API key

### Setup

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo-url>
   cd cannes
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` file with your OpenAI API key:**
   ```bash
   # OpenAI Configuration
   OPENAI_API_KEY=your_actual_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_TEMPERATURE=0.7
   ```

4. **Start all services:**
   ```bash
   ./start-services.sh
   ```

   Or manually:
   ```bash
   docker-compose up -d
   ```

## 📊 Services Overview

### ETL Service (Port 3030)
- **Endpoint:** `http://localhost:3030`
- **Purpose:** File upload and initial data processing
- **Supported formats:** CSV, Excel, JSON

### Preprocessing Service (Port 3031)
- **Endpoint:** `http://localhost:3031`
- **Purpose:** Data cleaning, normalization, and ML preparation
- **Features:** Missing value handling, feature engineering

### EDA Service (Port 3035)
- **Endpoint:** `http://localhost:3035`
- **Purpose:** Exploratory data analysis and visualization
- **Features:** Statistical analysis, interactive charts, insights generation

### Analysis Service (Port 3040)
- **Endpoint:** `http://localhost:3040`
- **Purpose:** Machine learning analysis with TEE attestation
- **Features:** ML algorithms, model evaluation, cryptographic attestation

## 🔧 Management Commands

```bash
# View service logs
docker-compose logs -f

# Stop all services
docker-compose down

# Restart services
docker-compose restart

# Check service status
docker-compose ps

# Pull latest images
docker-compose pull
```

## 🌐 Web Interface

The pipeline works with your Next.js web application. Make sure to:

1. Start the Docker services (as described above)
2. Run your Next.js application:
   ```bash
   npm run dev
   # or
   yarn dev
   ```
3. Navigate to your analyze page to upload and analyze data

## 📁 Directory Structure

```
cannes/
├── docker-compose.yml          # Master compose file
├── .env.example               # Environment variables template
├── .env                       # Your actual environment variables
├── start-services.sh          # Service startup script
├── data/                      # Data storage directory
├── results/                   # Analysis results directory
├── pages/
│   ├── analyze.js            # Analysis web interface
│   └── upload.js             # File upload interface
├── analysis/
│   ├── app.py                # Analysis service with TEE
│   ├── tee_signing.py        # TEE attestation module
│   └── compose.yaml          # Individual service compose
├── eda/
│   └── compose.yaml          # EDA service compose
├── etl/
│   └── compose.yaml          # ETL service compose
└── preprocessing/
    └── compose.yaml          # Preprocessing service compose
```

## 🔐 TEE Attestation

The Analysis Service includes Trusted Execution Environment (TEE) attestation:

- **Integrity Protection:** Cryptographic hashing ensures data hasn't been tampered with
- **Authenticity:** Digital signatures prove the analysis origin
- **Non-repudiation:** ROFL app ID provides cryptographic proof of execution

## 🐛 Troubleshooting

### Services Not Starting
```bash
# Check Docker is running
docker info

# Check service logs
docker-compose logs [service-name]

# Restart specific service
docker-compose restart [service-name]
```

### Port Conflicts
If ports are already in use, modify the port mappings in `docker-compose.yml`:
```yaml
ports:
  - "3031:3030"  # Change left port number
```

### Environment Variables
Ensure your `.env` file has the correct OpenAI API key:
```bash
# Check current environment
docker-compose config
```

## 📈 Usage Flow

1. **Upload Data** → ETL Service processes the file
2. **Preprocessing** → Data is cleaned and prepared
3. **EDA Analysis** → Generates visualizations and insights
4. **ML Analysis** → Runs machine learning algorithms with TEE attestation
5. **Results** → View comprehensive analysis with cryptographic verification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the local Docker setup
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Analyzing! 🎉**
