import Link from "next/link";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { uploadFile } from "../utils/writeToWalrus";
import { getUserEncryptionKey, createEncryptedFile, uint8ArrayToBase64 } from "../utils/encryption";
import WalletConnect from '../components/WalletConnect';
import dynamic from 'next/dynamic';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function Upload() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isUploadComplete, setIsUploadComplete] = useState(false);
  const [uploadedBlobId, setUploadedBlobId] = useState(null);
  const [uploadError, setUploadError] = useState('');
  
  // Analysis functionality
  const [analysisGoal, setAnalysisGoal] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState('');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [analysisError, setAnalysisError] = useState('');
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);
  const [serviceStatus, setServiceStatus] = useState({});
  const [originalFileForAnalysis, setOriginalFileForAnalysis] = useState(null);
  // Add new state for AI insights loading
  const [aiInsightsReady, setAiInsightsReady] = useState(false);

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      setSelectedFiles(files);
      setIsUploadComplete(false);
      setUploadedBlobId(null);
      setUploadError('');
      setUploadProgress(0);
      setOriginalFileForAnalysis(null); // Clear previous file
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const files = Array.from(event.dataTransfer.files);
    if (files.length > 0) {
      setSelectedFiles(files);
      setIsUploadComplete(false);
      setUploadedBlobId(null);
      setUploadError('');
      setUploadProgress(0);
      setOriginalFileForAnalysis(null); // Clear previous file
    }
  };

  const handleUpload = async () => {
    if (!selectedFiles.length) return;
    setIsUploading(true);
    setUploadProgress(0);
    setIsUploadComplete(false);
    setUploadError('');
    setUploadedBlobId(null);
    let completed = 0;
    try {
      // Get user's encryption key
      const encryptionKey = await getUserEncryptionKey();
      
      for (const file of selectedFiles) {
        // Simulate upload progress for UI feedback
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => {
            if (prev >= 90) {
              clearInterval(progressInterval);
              return 90;
            }
            return prev + 15;
          });
        }, 100);
        
        // Encrypt the file before upload
        const { encryptedFile, iv, metadata } = await createEncryptedFile(file, encryptionKey);
        
        // Upload encrypted file to Walrus
        const result = await uploadFile(encryptedFile, {
          epochs: 1,
          deletable: true
        });
        
        clearInterval(progressInterval);
        setUploadProgress(100);
        
        // Store file metadata in localStorage for view.js (including encryption info)
        const fileMetadata = {
          id: Date.now() + Math.random(),
          name: file.name, // Store original name
          size: formatFileSize(file.size), // Store original size
          type: file.type || 'Unknown', // Store original type
          blobId: result.blobId,
          timestamp: new Date().toISOString(),
          isActive: true,
          originalSize: file.size,
          // Encryption metadata
          isEncrypted: true,
          encryptionIV: uint8ArrayToBase64(iv), // Store IV as base64
          encryptionMetadata: metadata, // Store original file metadata
        };
        
        const existingFiles = JSON.parse(localStorage.getItem('walrusFiles') || '[]');
        const updatedFiles = [fileMetadata, ...existingFiles];
        localStorage.setItem('walrusFiles', JSON.stringify(updatedFiles));
        setUploadedBlobId(result.blobId);
        completed++;
      }
      setIsUploadComplete(true);
      setIsUploading(false);
      setUploadProgress(0);
      setSelectedFiles([]);
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadError(error.message || 'Upload failed. Please try again.');
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Improved fetch with timeout and better error handling (from analyze.js)
  const fetchWithTimeout = async (url, options = {}, timeout = 30000) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new Error('Request timed out');
      }
      throw error;
    }
  };

  // Check individual service health
  const checkServiceHealth = async (serviceName, url) => {
    try {
      const response = await fetch(url, {
        method: 'GET',
        timeout: 5000
      });
      const data = await response.json();
      return { healthy: true, status: data.status };
    } catch (err) {
      return { healthy: false, error: err.message };
    }
  };

  // Retry mechanism for failed requests (from analyze.js)
  const fetchWithRetry = async (url, options = {}, timeout = 30000, maxRetries = 2) => {
    let lastError;
    
    for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
      try {
        if (attempt > 1) {
          console.log(`Retry attempt ${attempt - 1}/${maxRetries} for ${url}`);
          // Wait before retry (exponential backoff)
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt - 1) * 1000));
        }
        
        return await fetchWithTimeout(url, options, timeout);
      } catch (error) {
        lastError = error;
        console.log(`Attempt ${attempt} failed: ${error.message}`);
        
        // If connection was reset or empty response, check service health
        if (error.message.includes('Failed to fetch') || error.message.includes('ERR_CONNECTION_RESET') || error.message.includes('ERR_EMPTY_RESPONSE')) {
          console.log(`Connection issue detected, checking service health...`);
          const serviceUrl = url.replace(/\/[^\/]*$/, ''); // Remove endpoint path
          const healthCheck = await checkServiceHealth('service', serviceUrl);
          
          if (!healthCheck.healthy) {
            console.log(`Service appears to be down: ${healthCheck.error}`);
            // Add longer wait for service recovery
            if (attempt < maxRetries + 1) {
              console.log(`Waiting 5 seconds for service recovery...`);
              await new Promise(resolve => setTimeout(resolve, 5000));
            }
          }
        }
        
        if (attempt === maxRetries + 1) {
          throw lastError;
        }
      }
    }
    
    throw lastError;
  };

  // Get helpful error message and recovery suggestions (from analyze.js)
  const getErrorMessage = (error, step) => {
    const baseMessage = error.message || 'Unknown error occurred';
    
    if (baseMessage.includes('Failed to fetch') || baseMessage.includes('ERR_CONNECTION_RESET') || baseMessage.includes('ERR_EMPTY_RESPONSE')) {
      return {
        message: `${step} service connection failed`,
        details: `The ${step.toLowerCase()} service appears to be temporarily unavailable. This can happen when processing large datasets.`,
        suggestions: [
          'Try with a smaller dataset first',
          'Wait a moment and try again',
          'Check if all Docker services are running',
          'Consider restarting the services if the problem persists'
        ]
      };
    }
    
    if (baseMessage.includes('timeout') || baseMessage.includes('Request timed out')) {
      return {
        message: `${step} service timed out`,
        details: `The ${step.toLowerCase()} service is taking too long to process your data.`,
        suggestions: [
          'Try with a smaller dataset',
          'The service may be processing a large dataset - please wait',
          'Consider increasing timeout settings'
        ]
      };
    }
    
    return {
      message: `${step} failed: ${baseMessage}`,
      details: 'An unexpected error occurred during processing.',
      suggestions: [
        'Try again with the same data',
        'Check the debug logs for more details',
        'Verify your data format is correct'
      ]
    };
  };

  // Check service health
  const checkServices = async () => {
    const services = {
      etl: 'http://localhost:3030',
      preprocessing: 'http://localhost:3031', 
      eda: 'http://localhost:3035',
      analysis: 'http://localhost:3040'
    };

    const status = {};
    for (const [name, url] of Object.entries(services)) {
      try {
        const response = await fetch(url, {
          method: 'GET',
          timeout: 5000
        });
        const data = await response.json();
        status[name] = { healthy: true, status: data.status };
      } catch (err) {
        status[name] = { healthy: false, error: err.message };
      }
    }
    setServiceStatus(status);
    return status;
  };

  // Helper function to validate AI insights readiness
  const validateAIInsightsReady = (edaData) => {
    console.log('=== AI Insights Validation Debug ===');
    console.log('Full EDA Data:', edaData);
    
    if (!edaData || !edaData.analysis) {
      console.log('❌ No analysis data found');
      return false;
    }
    
    console.log('Analysis data exists:', !!edaData.analysis);
    console.log('Insights data:', edaData.analysis.insights);
    
    // Check for insights in multiple possible locations
    const insights = edaData.analysis.insights;
    
    if (!insights) {
      console.log('❌ No insights object found');
      return false;
    }
    
    // Check different possible insight structures
    if (insights.ai_insights) {
      console.log('✅ Found ai_insights structure');
      const aiInsights = insights.ai_insights;
      console.log('AI Insights keys:', Object.keys(aiInsights));
      
      // Check if any category has insights
      for (const [category, data] of Object.entries(aiInsights)) {
        console.log(`Checking category: ${category}`, data);
        if (data && data.insights && Array.isArray(data.insights) && data.insights.length > 0) {
          console.log(`✅ Found valid insights in category: ${category}`);
          return true;
        }
      }
    }
    
    // Check for direct insights array
    if (Array.isArray(insights)) {
      console.log('✅ Found direct insights array');
      console.log('Insights array length:', insights.length);
      return insights.length > 0;
    }
    
    // Check for any insights property
    if (insights.insights && Array.isArray(insights.insights)) {
      console.log('✅ Found insights.insights array');
      console.log('Insights length:', insights.insights.length);
      return insights.insights.length > 0;
    }
    
    // More flexible check - if insights object exists and has any content
    if (typeof insights === 'object' && Object.keys(insights).length > 0) {
      console.log('✅ Found insights object with content, accepting as valid');
      return true;
    }
    
    console.log('❌ No valid insights found');
    return false;
  };

  // Analysis pipeline function (improved version from analyze.js)
  const runAnalysisPipeline = async (actualFile, goal) => {
    try {
      setAnalysisStep('Checking services...');
      setAiInsightsReady(false); // Reset AI insights state
      const services = await checkServices();
      
      // Check if all services are healthy
      const unhealthyServices = Object.entries(services).filter(([name, status]) => !status.healthy);
      if (unhealthyServices.length > 0) {
        throw new Error(`Services not available: ${unhealthyServices.map(([name]) => name).join(', ')}`);
      }

      // Create FormData for file upload (EXACTLY like analyze.js)
      const formData = new FormData();
      formData.append('file', actualFile);
      formData.append('goal', goal || 'data analysis');

      // Step 1: ETL Service (File Upload - EXACTLY like analyze.js)
      setAnalysisStep('Processing data...');
      console.log('Step 1: ETL Processing...');
      
      const etlResponse = await fetchWithRetry('http://localhost:3030/analyze', {
        method: 'POST',
        body: formData  // Use FormData with actual file like analyze.js
      }, 30000);
      
      const etlData = await etlResponse.json();
      console.log(`ETL Success: Data processed with ${etlData.processed_data ? etlData.processed_data.length : 'unknown'} rows`);

      // Step 2: Preprocessing Service
      setAnalysisStep('Preprocessing data...');
      console.log('Step 2: Preprocessing...');
      
      const preprocessResponse = await fetchWithRetry('http://localhost:3031/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: etlData.processed_data || etlData.data,
          goal: goal || 'machine learning preparation'
        })
      }, 30000);

      const preprocessData = await preprocessResponse.json();
      console.log(`Step 2 Success: Preprocessed ${preprocessData.processed_shape ? preprocessData.processed_shape[0] : 'unknown'} rows`);

      // Step 3: EDA Service
      setAnalysisStep('Generating visualizations...');
      console.log('Step 3: EDA Analysis...');
      
      const edaResponse = await fetchWithRetry('http://localhost:3035/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: preprocessData.processed_data || preprocessData.data,
          prompt: goal || 'Comprehensive data analysis'
        })
      }, 45000); // Longer timeout for EDA

      const edaData = await edaResponse.json();
      console.log(`Step 3 Success: Generated ${edaData.analysis && edaData.analysis.visualizations ? edaData.analysis.visualizations.length : 0} visualizations`);

      // Step 3.5: Wait for AI insights to be ready
      setAnalysisStep('Processing AI insights...');
      console.log('Step 3.5: Validating AI insights...');
      
      // Validate that AI insights are properly formatted and ready using helper function
      const insightsReady = validateAIInsightsReady(edaData);
      
      if (insightsReady) {
        console.log('AI insights validated and ready');
        setAiInsightsReady(true);
      } else {
        console.log('AI insights validation failed, but checking for any insights structure...');
        
        // Fallback: check if there's any insights structure at all
        if (edaData.analysis && edaData.analysis.insights) {
          console.log('Found insights structure, setting as ready anyway');
          setAiInsightsReady(true);
        } else {
          console.log('No insights structure found, continuing without insights');
          setAiInsightsReady(false);
        }
      }
      
      // Additional debug: log the current state
      console.log('AI Insights Ready State:', aiInsightsReady);
      console.log('Will show visualizations:', insightsReady || (edaData.analysis && edaData.analysis.insights));

      // Step 4: ML Analysis Service
      setAnalysisStep('Running machine learning analysis...');
      console.log('Step 4: ML Analysis...');
      
      const mlResponse = await fetchWithRetry('http://localhost:3040/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: preprocessData.processed_data || preprocessData.data,
          goal: goal || 'comprehensive analysis'
        })
      }, 60000); // Longer timeout for ML

      const mlData = await mlResponse.json();
      console.log('Step 4 Success: Completed ML analysis');

      // Final step: Ensure AI insights are fully ready
      setAnalysisStep('Finalizing AI insights...');
      await new Promise(resolve => setTimeout(resolve, 1000)); // Small delay to ensure everything is ready
      
      // Final validation and state update
      const finalInsightsCheck = edaData.analysis && edaData.analysis.insights;
      if (finalInsightsCheck) {
        console.log('Final check: Setting AI insights as ready');
        setAiInsightsReady(true);
      }
      
      // Combine all results
      const results = {
        etl: etlData,
        preprocessing: preprocessData,
        eda: edaData,
        ml: mlData,
        goal: goal,
        timestamp: new Date().toISOString(),
        aiInsightsReady: finalInsightsCheck // Use final check result
      };
      
      console.log('=== Full Pipeline Completed Successfully! ===');
      console.log('Final AI Insights Ready State:', finalInsightsCheck);
      return results;

    } catch (error) {
      const currentStepName = analysisStep.includes('Processing') ? 'ETL' :
                             analysisStep.includes('Preprocessing') ? 'Preprocessing' :
                             analysisStep.includes('visualizations') ? 'EDA' :
                             analysisStep.includes('AI insights') ? 'AI Insights' :
                             analysisStep.includes('machine learning') ? 'Analysis' : 'Unknown';
      
      const errorInfo = getErrorMessage(error, currentStepName);
      console.error(`Pipeline Error: ${errorInfo.message}`);
      console.error(`Details: ${errorInfo.details}`);
      errorInfo.suggestions.forEach(suggestion => {
        console.log(`💡 Suggestion: ${suggestion}`);
      });
      
      setAiInsightsReady(false);
      throw error;
    }
  };

  // Combined upload and analysis function
  const handleUploadAndAnalyze = async () => {
    if (!selectedFiles.length || !analysisGoal.trim()) return;
    
    setIsUploading(true);
    setUploadProgress(0);
    setIsUploadComplete(false);
    setUploadError('');
    setUploadedBlobId(null);
    setAnalysisError('');
    
    let completed = 0;
    let lastFileMetadata = null;
    
    // Store the original file for analysis
    setOriginalFileForAnalysis(selectedFiles[0]);
    
    try {
      // First upload the file
      for (const file of selectedFiles) {
        // Simulate upload progress for UI feedback
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => {
            if (prev >= 90) {
              clearInterval(progressInterval);
              return 90;
            }
            return prev + 15;
          });
        }, 100);
        
        // Upload file to Walrus
        const result = await uploadFile(file, {
          epochs: 1,
          deletable: true
        });
        
        clearInterval(progressInterval);
        setUploadProgress(100);
        
        // Store file metadata in localStorage
        const fileMetadata = {
          id: Date.now() + Math.random(),
          name: file.name,
          size: formatFileSize(file.size),
          type: file.type || 'Unknown',
          blobId: result.blobId,
          timestamp: new Date().toISOString(),
          isActive: true,
          originalSize: file.size
        };
        
        const existingFiles = JSON.parse(localStorage.getItem('walrusFiles') || '[]');
        const updatedFiles = [fileMetadata, ...existingFiles];
        localStorage.setItem('walrusFiles', JSON.stringify(updatedFiles));
        
        setUploadedBlobId(result.blobId);
        lastFileMetadata = fileMetadata; // Store reference for analysis
        completed++;
      }
      
      setIsUploadComplete(true);
      setIsUploading(false);
      setUploadProgress(0);
      
      // Now start the analysis with the file data
      if (lastFileMetadata) {
        await handleAnalyzeDataWithFile(lastFileMetadata, selectedFiles[0]);
      }
      
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadError(error.message || 'Upload failed. Please try again.');
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  // Render simplified insights (copied from analyze.js)
  const renderSimpleInsights = (insights) => {
    console.log('=== Rendering Insights Debug ===');
    console.log('Insights to render:', insights);
    
    if (!insights) {
      return (
        <div className="bg-white p-4 rounded-lg border">
          <p className="text-sm text-gray-600">🤖 No insights available yet. Analysis may still be processing.</p>
        </div>
      );
    }

    // Handle different insight structures
    let allInsights = [];
    let keyFindings = [];
    let recommendations = [];
    let actualInsights = [];

    // Check for ai_insights structure
    if (insights.ai_insights) {
      console.log('Processing ai_insights structure');
      const aiInsightsData = Object.values(insights.ai_insights);
      
      aiInsightsData.forEach((category, index) => {
        if (category && category.insights && Array.isArray(category.insights)) {
          category.insights.forEach(insightString => {
            if (typeof insightString === 'string') {
              try {
                // Remove markdown code block formatting if present
                let cleanJson = insightString;
                if (cleanJson.includes('```json')) {
                  cleanJson = cleanJson.replace(/^```json\s*/, '').replace(/\s*```$/, '').trim();
                }
                
                // Parse the JSON string
                const parsedInsight = JSON.parse(cleanJson);
                
                // Extract data from parsed object
                if (parsedInsight.insights && Array.isArray(parsedInsight.insights)) {
                  actualInsights.push(...parsedInsight.insights);
                }
                
                if (parsedInsight.key_findings && Array.isArray(parsedInsight.key_findings)) {
                  keyFindings.push(...parsedInsight.key_findings);
                }
                
                if (parsedInsight.recommendations && Array.isArray(parsedInsight.recommendations)) {
                  recommendations.push(...parsedInsight.recommendations);
                }
              } catch (e) {
                console.warn('Could not parse insight JSON:', insightString);
                // If parsing fails, treat as plain text insight
                if (insightString.length > 10) {
                  actualInsights.push(insightString);
                }
              }
            }
          });
        }
      });
    }

    // Check for direct insights array
    if (Array.isArray(insights)) {
      console.log('Processing direct insights array');
      actualInsights.push(...insights);
    }

    // Check for insights.insights
    if (insights.insights && Array.isArray(insights.insights)) {
      console.log('Processing insights.insights array');
      actualInsights.push(...insights.insights);
    }

    // Check for any other structure and extract text
    if (typeof insights === 'object' && !Array.isArray(insights) && !insights.ai_insights) {
      console.log('Processing generic insights object');
      Object.values(insights).forEach(value => {
        if (typeof value === 'string' && value.length > 10) {
          actualInsights.push(value);
        } else if (Array.isArray(value)) {
          actualInsights.push(...value.filter(item => typeof item === 'string' && item.length > 10));
        }
      });
    }

    // Filter and deduplicate meaningful content
    const getUniqueFiltered = (arr) => {
      const filtered = arr.filter(item => 
        item && 
        typeof item === 'string' &&
        item !== "AI analysis completed" && 
        !item.includes("Review the insights provided") &&
        item.length > 10
      );
      // Remove duplicates
      return [...new Set(filtered)];
    };

    const filteredInsights = getUniqueFiltered(actualInsights);
    const filteredFindings = getUniqueFiltered(keyFindings);
    const filteredRecommendations = getUniqueFiltered(recommendations);

    console.log('Filtered insights:', filteredInsights);
    console.log('Filtered findings:', filteredFindings);
    console.log('Filtered recommendations:', filteredRecommendations);

    // If no structured insights found, show raw data
    if (filteredInsights.length === 0 && filteredFindings.length === 0 && filteredRecommendations.length === 0) {
      return (
        <div className="space-y-4">
          <div className="bg-white p-4 rounded-lg border">
            <h4 className="font-medium text-blue-900 mb-3">🧠 Raw Analysis Data</h4>
            <div className="text-sm text-gray-700">
              <pre className="whitespace-pre-wrap bg-gray-50 p-3 rounded text-xs max-h-40 overflow-y-auto">
                {JSON.stringify(insights, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {filteredInsights.length > 0 && (
          <div className="bg-white p-4 rounded-lg border">
            <h4 className="font-medium text-blue-900 mb-3">🧠 AI Insights</h4>
            <ul className="list-disc list-inside text-sm text-gray-700 space-y-2">
              {filteredInsights.slice(0, 8).map((insight, idx) => (
                <li key={idx}>{insight}</li>
              ))}
            </ul>
          </div>
        )}

        {filteredFindings.length > 0 && (
          <div className="bg-white p-4 rounded-lg border">
            <h4 className="font-medium text-blue-900 mb-3">🔍 Key Findings</h4>
            <ul className="list-disc list-inside text-sm text-gray-700 space-y-2">
              {filteredFindings.slice(0, 6).map((finding, idx) => (
                <li key={idx}>{finding}</li>
              ))}
            </ul>
          </div>
        )}
        
        {filteredRecommendations.length > 0 && (
          <div className="bg-white p-4 rounded-lg border">
            <h4 className="font-medium text-blue-900 mb-3">💡 Recommendations</h4>
            <ul className="list-disc list-inside text-sm text-gray-700 space-y-2">
              {filteredRecommendations.slice(0, 6).map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </div>
        )}

        {filteredInsights.length === 0 && filteredFindings.length === 0 && filteredRecommendations.length === 0 && (
          <div className="bg-white p-4 rounded-lg border">
            <p className="text-sm text-gray-600">🤖 Analysis in progress - detailed insights will appear here once processing is complete.</p>
          </div>
        )}
      </div>
    );
  };

  // Render TEE attestation information (copied from analyze.js)
  const renderTEEAttestation = (mlData) => {
    if (!mlData || !mlData.results || !mlData.results.tee_attestation) return null;

    const attestation = mlData.results.tee_attestation;
    
    return (
      <div className="bg-green-50 p-4 rounded-lg border border-green-200 mb-6">
        <h4 className="font-medium text-green-900 mb-3 flex items-center gap-2">
          🔐 TEE Attestation
          {attestation.tee_attested ? (
            <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">Verified</span>
          ) : (
            <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">Unavailable</span>
          )}
        </h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            {attestation.tee_attested ? (
              <div className="flex items-start gap-3 text-green-700">
                <span className="text-xl">✅</span>
                <div>
                  <strong className="block mb-1">TEE Verified</strong>
                  <p className="text-sm">This analysis was executed and signed in a Trusted Execution Environment</p>
                </div>
              </div>
            ) : (
              <div className="flex items-start gap-3 text-yellow-700">
                <span className="text-xl">⚠️</span>
                <div>
                  <strong className="block mb-1">TEE Unavailable</strong>
                  <p className="text-sm">Analysis completed but TEE attestation failed: {attestation.error}</p>
                </div>
              </div>
            )}
          </div>
          
          {attestation.tee_attested && (
            <div className="space-y-2">
              <div className="bg-white p-3 rounded border">
                <div className="grid grid-cols-1 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span className="font-medium text-gray-600">ROFL App ID:</span>
                    <code className="bg-gray-100 px-1 rounded text-gray-800">{attestation.rofl_app_id}</code>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium text-gray-600">Results Hash:</span>
                    <code className="bg-gray-100 px-1 rounded text-gray-800">{attestation.results_hash?.substring(0, 16)}...</code>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium text-gray-600">Algorithm:</span>
                    <span className="text-gray-800">{attestation.signature_algorithm}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium text-gray-600">Timestamp:</span>
                    <span className="text-gray-800">{new Date(attestation.timestamp).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Handle analysis with file data passed directly
  const handleAnalyzeDataWithFile = async (fileMetadata, originalFile) => {
    if (!analysisGoal.trim()) {
      setAnalysisError('Please enter an analysis goal');
      return;
    }

    setShowAnalysisModal(true);
    setIsAnalyzing(true);
    setAnalysisError('');
    setAnalysisResults(null);
    setAiInsightsReady(false); // Reset AI insights state

    try {
      // Use the actual original file for analysis (EXACTLY like analyze.js)
      const results = await runAnalysisPipeline(originalFile, analysisGoal);
      
      // Store analysis results on Walrus (to avoid browser storage limits)
      console.log('Storing analysis results on Walrus...');
      const analysisBlob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
      const analysisFile = new File([analysisBlob], `analysis-${fileMetadata.name}-${Date.now()}.json`, { type: 'application/json' });
      
      const analysisUploadResult = await uploadFile(analysisFile, {
        epochs: 1,
        deletable: true
      });

      console.log(`Analysis results stored on Walrus with ID: ${analysisUploadResult.blobId}`);

      // Store analysis metadata with the file (but not full results to save space)
      const updatedFile = {
        ...fileMetadata,
        analysisGoal: analysisGoal,
        hasAnalysis: true,
        lastAnalyzed: new Date().toISOString(),
        analysisResultsBlobId: analysisUploadResult.blobId // Reference to Walrus-stored results
      };

      // Update localStorage with analysis metadata only
      const storedFiles = JSON.parse(localStorage.getItem('walrusFiles') || '[]');
      const updatedFiles = storedFiles.map(file => 
        file.blobId === fileMetadata.blobId ? updatedFile : file
      );

      localStorage.setItem('walrusFiles', JSON.stringify(updatedFiles));
      
      // Store only metadata in localStorage (much smaller)
      const analysisReport = {
        id: Date.now() + Math.random(),
        fileName: fileMetadata.name,
        fileBlobId: fileMetadata.blobId,
        analysisGoal: analysisGoal,
        analysisResultsBlobId: analysisUploadResult.blobId, // Reference to Walrus-stored results
        timestamp: new Date().toISOString(),
        status: 'completed',
        resultSize: Math.round(analysisBlob.size / 1024) + ' KB', // Show size for reference
        isActive: true // Default to active for new reports
      };

      const existingReports = JSON.parse(localStorage.getItem('analysisReports') || '[]');
      const updatedReports = [analysisReport, ...existingReports];
      localStorage.setItem('analysisReports', JSON.stringify(updatedReports));
      
      setAnalysisResults(results);
      setAnalysisStep('Analysis complete! Results stored on Walrus.');
      
      // Force update AI insights state after setting results
      setTimeout(() => {
        if (results.eda?.analysis?.insights) {
          console.log('Force updating AI insights state to ready');
          setAiInsightsReady(true);
        }
      }, 500);
      
      // Keep modal open to show results instead of auto-closing
      setIsAnalyzing(false);

    } catch (error) {
      setAnalysisError(error.message);
      setIsAnalyzing(false);
      setAiInsightsReady(false); // Reset on error
    }
  };

  // Handle analysis (updated to work with uploaded file)
  const handleAnalyzeData = async () => {
    if (!analysisGoal.trim()) {
      setAnalysisError('Please enter an analysis goal');
      return;
    }

    setShowAnalysisModal(true);
    setIsAnalyzing(true);
    setAnalysisError('');
    setAnalysisResults(null);
    setAiInsightsReady(false); // Reset AI insights state

    try {
      // Get the uploaded file data from localStorage
      const storedFiles = JSON.parse(localStorage.getItem('walrusFiles') || '[]');
      const uploadedFile = storedFiles.find(file => file.blobId === uploadedBlobId);
      
      if (!uploadedFile) {
        throw new Error('Uploaded file not found');
      }

      if (!originalFileForAnalysis) {
        throw new Error('Original file not available for re-analysis');
      }

      await handleAnalyzeDataWithFile(uploadedFile, originalFileForAnalysis);

    } catch (error) {
      setAnalysisError(error.message);
      setIsAnalyzing(false);
      setAiInsightsReady(false); // Reset on error
    }
  };

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.4,
        ease: "easeOut",
      },
    },
  };

  return (
    <div className="h-screen font-montserrat bg-gradient-to-br from-blue-50 to-indigo-100 text-slate-900 transition-colors duration-300 overflow-hidden flex flex-col">
      {/* Navigation Bar */}
      <motion.nav 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="relative flex justify-center pt-8 pb-4 px-8 flex-shrink-0"
      >
        <div className="flex bg-white/80 backdrop-blur-sm rounded-full p-1 transition-all duration-300 shadow-lg hover:shadow-xl border border-blue-200">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full hover:bg-blue-100 text-slate-700 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Chat
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full bg-blue-600 text-white font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95 shadow-md">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full hover:bg-blue-100 text-slate-700 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              View
            </div>
          </Link>
          <Link href="/subscribe">
            <div className="px-6 py-2 rounded-full hover:bg-blue-100 text-slate-700 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Subscribe
            </div>
          </Link>
        </div>
        <div className="absolute right-8 top-8">
          <WalletConnect />
        </div>
      </motion.nav>

      {/* Page Header */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut", delay: 0.1 }}
        className="px-8 py-4 flex-shrink-0"
      >
        <h1 className="text-3xl font-bold text-center text-slate-800 transform transition-all duration-300">Upload Your Data</h1>
        <p className="text-center text-slate-600 mt-2 transition-opacity duration-200">
          Upload your datasets to start analyzing and discovering insights
        </p>
      </motion.div>

      {/* Main Content */}
      <motion.main 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="flex-1 flex flex-col items-center px-8 min-h-0 overflow-y-auto justify-start pt-8"
      >
        <div className="max-w-2xl w-full">
          
          {/* File Upload Area */}
          <motion.div
            variants={itemVariants}
            className={`border-2 rounded-2xl p-12 text-center transition-all duration-300 transform hover:scale-[1.02] ${
              selectedFiles.length > 0
                ? 'border-solid border-emerald-400 bg-emerald-50/80 backdrop-blur-sm scale-[1.02] shadow-lg'
                : isDragging
                ? 'border-dashed border-blue-500 bg-blue-100/80 backdrop-blur-sm scale-[1.05] shadow-2xl animate-pulse'
                : 'border-dashed border-blue-300 hover:border-blue-400 bg-white/80 backdrop-blur-sm shadow-md hover:shadow-xl'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="space-y-4">
              <div className="flex justify-center">
                {selectedFiles.length > 0 ? (
                  <svg className="w-12 h-12 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                ) : (
                  <svg
                    className={`w-12 h-12 text-slate-600 transition-all duration-300 ${isDragging ? 'animate-bounce text-blue-700' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                )}
              </div>
              
              <div>
                {selectedFiles.length > 0 ? (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <p className="text-lg font-medium mb-2 text-emerald-700">
                      {selectedFiles.map(file => file.name).join(', ')}
                    </p>
                    <p className="text-sm text-black">
                      {formatFileSize(selectedFiles[0].size)} • Ready to upload
                    </p>
                  </motion.div>
                ) : (
                  <div>
                    <p className="text-lg font-medium mb-2 text-slate-800 transition-all duration-200">
                      Drop your files here
                    </p>
                    <p className="text-slate-600">
                      or click to browse
                    </p>
                  </div>
                )}
              </div>
              
              <input
                type="file"
                onChange={handleFileSelect}
                className="hidden"
                id="fileInput"
                accept=".csv,.xlsx,.xls,.json,.txt"
                multiple
              />
              
              {!selectedFiles.length && (
                <label
                  htmlFor="fileInput"
                  className="inline-block px-6 py-3 bg-blue-200 text-slate-800 rounded-lg font-medium cursor-pointer hover:bg-blue-300 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                >
                  Select Files
                </label>
              )}

              {selectedFiles.length > 0 && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className="flex gap-2 justify-center"
                >
                  <label
                    htmlFor="fileInput"
                    className="px-4 py-2 bg-blue-200 text-slate-800 rounded-lg font-medium cursor-pointer hover:bg-blue-300 transition-all duration-200 text-sm transform hover:scale-105 active:scale-95 shadow-sm hover:shadow-md"
                  >
                    Change Files
                  </label>
                  <button
                    onClick={() => {
                      setSelectedFiles([]);
                      setIsUploadComplete(false);
                      setUploadedBlobId(null);
                      setUploadError('');
                      setUploadProgress(0);
                      setOriginalFileForAnalysis(null);
                    }}
                    className="px-4 py-2 bg-rose-100 text-rose-700 rounded-lg font-medium hover:bg-rose-200 transition-all duration-200 text-sm transform hover:scale-105 active:scale-95 shadow-sm hover:shadow-md"
                  >
                    Remove All
                  </button>
                </motion.div>
              )}
            </div>
          </motion.div>



          {/* Analysis Goal Input */}
          {selectedFiles.length > 0 && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="mt-6 bg-white/80 backdrop-blur-sm p-6 rounded-2xl border border-blue-200 transition-all duration-300 shadow-md hover:shadow-lg"
            >
              <h3 className="font-semibold text-lg mb-4 text-black transform transition-all duration-200">Analysis Settings</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-black mb-2">
                    Analysis Goal
                  </label>
                  <textarea
                    value={analysisGoal}
                    onChange={(e) => setAnalysisGoal(e.target.value)}
                    placeholder="Enter your analysis goals (e.g., 'Find patterns in sales data', 'Predict customer behavior', 'Identify anomalies')"
                    rows={3}
                    className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent transition-all duration-300 text-black placeholder-gray-500 resize-none"
                  />
                </div>
                
                {analysisError && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="p-3 bg-red-50 border border-red-200 rounded-lg"
                  >
                    <p className="text-red-700 text-sm">{analysisError}</p>
                  </motion.div>
                )}
                
                <button
                  onClick={handleUploadAndAnalyze}
                  disabled={!analysisGoal.trim() || isUploading || isAnalyzing}
                  className={`w-full py-3 rounded-lg font-medium transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl ${
                    analysisGoal.trim() && !isUploading && !isAnalyzing
                      ? 'bg-gray-600 text-white hover:bg-gray-700 hover:-translate-y-1'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed scale-100'
                  }`}
                >
                  {isUploading ? (
                    <div className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Uploading...
                </div>
                  ) : isAnalyzing ? (
                    <div className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Analyzing...
                </div>
                  ) : (
                    'Upload & Analyze Data'
                  )}
                </button>
              </div>
            </motion.div>
          )}

          {/* Upload Progress */}
          {isUploading && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
              className="mt-6 bg-white/80 backdrop-blur-sm p-6 rounded-2xl border border-blue-200 shadow-lg"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-800">Uploading...</span>
                <span className="text-sm text-slate-700 animate-pulse">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-blue-200 rounded-full h-2 overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${uploadProgress}%` }}
                  transition={{ duration: 0.3 }}
                  className="bg-blue-600 h-2 rounded-full animate-pulse"
                ></motion.div>
              </div>
            </motion.div>
          )}

          {/* Error Message */}
          {uploadError && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="mt-6 bg-rose-50 p-6 rounded-2xl border border-rose-200 shadow-lg"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <svg className="w-6 h-6 text-rose-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-rose-700 font-medium">{uploadError}</span>
                </div>
                <button
                  onClick={() => setUploadError('')}
                  className="text-rose-500 hover:text-rose-700 ml-2"
                >
                  ×
                </button>
              </div>
            </motion.div>
          )}

          {/* Success Message */}
          {isUploadComplete && uploadedBlobId && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="mt-6 bg-emerald-50 p-4 rounded-2xl border border-emerald-200 shadow-lg"
            >
              <div className="text-center">
                <div className="flex items-center justify-center">
                  <svg className="w-6 h-6 text-emerald-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-emerald-700 font-medium">Upload completed successfully!</span>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </motion.main>
          
      {/* Analysis Loading Modal */}
      <AnimatePresence>
        {showAnalysisModal && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={() => !isAnalyzing && setShowAnalysisModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="p-6 border-b border-gray-200 bg-gray-50">
                <h3 className="text-lg font-semibold text-black text-center">
                  {isAnalyzing ? 'Analyzing Your Data' : analysisResults ? 'Analysis Complete!' : 'Analysis Failed'}
                </h3>
              </div>

              {/* Modal Content */}
              <div className="bg-white text-center">
                {isAnalyzing ? (
                  <div className="p-8 space-y-6">
                    {/* Rotating Sandglass Animation */}
                    <div className="flex justify-center">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        className="w-16 h-16"
                      >
                        <svg 
                          className="w-full h-full text-gray-600" 
                          fill="currentColor" 
                          viewBox="0 0 24 24"
                        >
                          <path d="M6 2V8H6.01L6 8.01L10.5 12L6 15.99L6.01 16H6V22H18V16H17.99L18 15.99L13.5 12L18 8.01L17.99 8H18V2H6ZM16 4V6.5L12 10.5L8 6.5V4H16ZM8 17.5L12 13.5L16 17.5V20H8V17.5Z"/>
                  </svg>
                      </motion.div>
                    </div>
                    
                    {/* Progress Text */}
                    <div className="space-y-2">
                      <p className="text-gray-700 font-medium">
                        {analysisStep || 'Preparing analysis...'}
                      </p>
                      
                      {/* Special indicator for AI insights processing */}
                      {analysisStep && (analysisStep.includes('AI insights') || analysisStep.includes('Finalizing')) && (
                        <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                          <p className="text-blue-800 text-sm">
                            🧠 AI insights are being processed and validated. This ensures high-quality analysis results.
                          </p>
                        </div>
                      )}
                      
                      <div className="flex justify-center">
                        <div className="flex space-x-1">
                          <motion.div 
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 1, repeat: Infinity, delay: 0 }}
                            className="w-2 h-2 bg-gray-600 rounded-full"
                          />
                          <motion.div 
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                            className="w-2 h-2 bg-gray-600 rounded-full"
                          />
                          <motion.div 
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                            className="w-2 h-2 bg-gray-600 rounded-full"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ) : analysisResults ? (
                  <div className="max-h-[70vh] overflow-y-auto text-left p-6">
                    {/* Analysis Results Content */}
                    <div className="space-y-8">
                      
                      {/* Header */}
                      <div className="text-center border-b pb-4">
                        <h2 className="text-2xl font-bold text-gray-800 mb-2">📊 Exploratory Data Analysis</h2>
                        <p className="text-gray-600">Comprehensive analysis results for your dataset</p>
                        <div className="mt-2 text-sm text-gray-500">
                          Goal: {analysisGoal}
                        </div>
                      </div>

                      {/* AI Insights */}
                      {analysisResults.eda?.analysis?.insights && (
                        <div className="space-y-4">
                          <h3 className="text-xl font-semibold text-gray-800 border-b pb-2">🧠 AI Insights</h3>
                          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                            {aiInsightsReady ? (
                              renderSimpleInsights(analysisResults.eda.analysis.insights)
                            ) : (
                              <div className="text-center py-8">
                                <motion.div
                                  animate={{ rotate: 360 }}
                                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                  className="w-8 h-8 mx-auto mb-4"
                                >
                                  <svg 
                                    className="w-full h-full text-blue-600" 
                                    fill="currentColor" 
                                    viewBox="0 0 24 24"
                                  >
                                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                                  </svg>
                                </motion.div>
                                <p className="text-blue-800 font-medium">Processing AI insights...</p>
                                <p className="text-blue-600 text-sm mt-1">
                                  Insights are being generated and will appear shortly
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* EDA Visualizations - Show regardless of AI insights status */}
                      {analysisResults.eda?.analysis?.visualizations && (
                        <div className="space-y-6">
                          <h3 className="text-xl font-semibold text-gray-800 border-b pb-2">📈 Data Visualizations</h3>
                          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                            {analysisResults.eda.analysis.visualizations.map((chart, index) => (
                              <div key={index} className="bg-gray-50 p-4 rounded-lg border">
                                <div className="mb-3">
                                  <h4 className="font-medium text-gray-800">{chart.title}</h4>
                                  <p className="text-sm text-gray-600">{chart.description}</p>
                                  <div className="flex gap-2 mt-2">
                                    <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                                      {chart.category}
                                    </span>
                                    <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded">
                                      {chart.chart_type}
                                    </span>
                                  </div>
                                </div>
                                
                                {chart.chart_json && Plot && (
                                  <div className="chart-container bg-white rounded border" style={{ width: '100%', height: '400px' }}>
                                    <Plot
                                      data={JSON.parse(chart.chart_json).data}
                                      layout={{
                                        ...JSON.parse(chart.chart_json).layout,
                                        width: undefined,
                                        height: undefined,
                                        autosize: true,
                                        margin: { l: 60, r: 40, t: 40, b: 60 },
                                        font: { size: 12 }
                                      }}
                                      config={{ 
                                        displayModeBar: true,
                                        responsive: true,
                                        displaylogo: false,
                                        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                                      }}
                                      style={{ width: '100%', height: '100%' }}
                                      useResizeHandler={true}
                                    />
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Machine Learning Results */}
                      {analysisResults.ml?.results?.analyses && (
                        <div className="space-y-6">
                          <h3 className="text-xl font-semibold text-gray-800 border-b pb-2">🤖 Machine Learning Analysis</h3>
                          
                          {/* TEE Attestation */}
                          {renderTEEAttestation(analysisResults.ml)}

                          {/* ML Analysis Results */}
                          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                            {analysisResults.ml.results.analyses.map((analysis, idx) => (
                              <div key={idx} className="bg-gray-50 p-4 rounded-lg border">
                                <h4 className="font-medium text-gray-800 mb-2">
                                  {analysis.algorithm} - {analysis.analysis_type}
                                </h4>
                                
                                {analysis.insights && (
                                  <div className="mb-3">
                                    <p className="text-sm font-medium text-gray-700 mb-1">Insights:</p>
                                    <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                                      {analysis.insights.slice(0, 3).map((insight, insightIdx) => (
                                        <li key={insightIdx}>{insight}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}

                                {analysis.graphs && analysis.graphs.length > 0 && (
                                  <div className="space-y-3">
                                    {analysis.graphs.filter(graph => !graph.title?.toLowerCase().includes('solution distribution')).slice(0, 1).map((graph, graphIdx) => (
                                      <div key={graphIdx} className="bg-white p-2 rounded border">
                                        <p className="text-xs font-medium text-gray-700 mb-2">{graph.title}</p>
                                        {graph.data && Plot && (
                                          <div style={{ width: '100%', height: '300px' }}>
                                            <Plot
                                              data={JSON.parse(graph.data).data}
                                              layout={{
                                                ...JSON.parse(graph.data).layout,
                                                width: undefined,
                                                height: undefined,
                                                autosize: true,
                                                margin: { l: 50, r: 30, t: 30, b: 40 },
                                                font: { size: 11 }
                                              }}
                                              config={{ 
                                                displayModeBar: true,
                                                responsive: true,
                                                displaylogo: false,
                                                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                                              }}
                                              style={{ width: '100%', height: '100%' }}
                                              useResizeHandler={true}
                                            />
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Pipeline Summary */}
                      <div className="bg-gray-50 p-4 rounded-lg border">
                        <h3 className="text-lg font-semibold text-gray-800 mb-3">📋 Analysis Pipeline Summary</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                          <div className="text-center p-3 bg-white rounded border">
                            <div className="text-lg font-bold text-green-600">
                              {analysisResults.etl ? '✓' : '✗'}
                            </div>
                            <div className="text-xs text-gray-600">ETL Processing</div>
                          </div>
                          <div className="text-center p-3 bg-white rounded border">
                            <div className="text-lg font-bold text-green-600">
                              {analysisResults.preprocessing ? '✓' : '✗'}
                            </div>
                            <div className="text-xs text-gray-600">Preprocessing</div>
                          </div>
                          <div className="text-center p-3 bg-white rounded border">
                            <div className="text-lg font-bold text-green-600">
                              {analysisResults.eda ? '✓' : '✗'}
                            </div>
                            <div className="text-xs text-gray-600">EDA Analysis</div>
                          </div>
                          <div className="text-center p-3 bg-white rounded border">
                            <div className="text-lg font-bold text-green-600">
                              {analysisResults.ml ? '✓' : '✗'}
                            </div>
                            <div className="text-xs text-gray-600">ML Analysis</div>
                          </div>
                        </div>
                      </div>
                    </div>
                </div>
              ) : (
                  <div className="p-8 space-y-4">
                    {/* Error Icon */}
                    <div className="flex justify-center">
                      <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
                        <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                      </div>
                    </div>
                    
                    <div>
                      <p className="text-red-700 font-medium mb-2">Analysis Failed</p>
                      <p className="text-sm text-gray-600">{analysisError}</p>
                    </div>
                    
                    <div className="flex gap-3 justify-center mt-6">
                      <button
                        onClick={() => setShowAnalysisModal(false)}
                        className="px-4 py-2 bg-gray-200 text-black rounded-lg hover:bg-gray-300 transition-all duration-200"
                      >
                        Close
                      </button>
                      <button
                        onClick={() => {
                          setAnalysisError('');
                          handleAnalyzeData();
                        }}
                        className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-all duration-200"
                      >
                        Retry
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Modal Footer */}
              {!isAnalyzing && analysisResults && (
                <div className="p-6 border-t border-gray-200 bg-gray-50 text-center space-x-3">
                  <button
                    onClick={() => {
                      setShowAnalysisModal(false);
                      // Reset form
                      setSelectedFiles([]);
                      setIsUploadComplete(false);
                      setUploadedBlobId(null);
                      setAnalysisGoal('');
                      setUploadProgress(0);
                      setAnalysisResults(null);
                      setOriginalFileForAnalysis(null);
                      setAiInsightsReady(false); // Reset AI insights state
                    }}
                    className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                  >
                    Complete Analysis
                  </button>
                  <button
                    onClick={() => {
                      const dataStr = JSON.stringify(analysisResults, null, 2);
                      const dataBlob = new Blob([dataStr], {type: 'application/json'});
                      const url = URL.createObjectURL(dataBlob);
                      const link = document.createElement('a');
                      link.href = url;
                      link.download = `analysis-results-${new Date().toISOString().split('T')[0]}.json`;
                      link.click();
                      URL.revokeObjectURL(url);
                    }}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                  >
                    Download Results
                  </button>
                </div>
              )}
          </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
