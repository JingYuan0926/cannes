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
