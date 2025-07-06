import Link from "next/link";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { uploadFile } from "../utils/writeToWalrus";
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

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      setSelectedFiles(files);
      setIsUploadComplete(false);
      setUploadedBlobId(null);
      setUploadError('');
      setUploadProgress(0);
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
        // Store file metadata in localStorage for view.js
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
        const response = await fetch(url);
        const data = await response.json();
        status[name] = { healthy: true, status: data.status };
      } catch (err) {
        status[name] = { healthy: false, error: err.message };
      }
    }
    setServiceStatus(status);
    return status;
  };

  // Analysis pipeline function
  const runAnalysisPipeline = async (fileData, goal) => {
    try {
      setAnalysisStep('Checking services...');
      const services = await checkServices();
      
      // Check if all services are healthy
      const unhealthyServices = Object.entries(services).filter(([name, status]) => !status.healthy);
      if (unhealthyServices.length > 0) {
        throw new Error(`Services not available: ${unhealthyServices.map(([name]) => name).join(', ')}`);
      }

      // Create FormData for file upload to analysis pipeline
      const formData = new FormData();
      // We need to reconstruct the file from the stored data
      // For now, we'll use the JSON data approach since we have the file content
      
      // Step 1: ETL Service
      setAnalysisStep('Processing data...');
      const etlResponse = await fetch('http://localhost:3030/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: fileData.analysisData || [],
          goal: goal || 'data analysis'
        })
      });
      
      if (!etlResponse.ok) {
        throw new Error('ETL processing failed');
      }
      
      const etlData = await etlResponse.json();

      // Step 2: Preprocessing Service
      setAnalysisStep('Preprocessing data...');
      const preprocessResponse = await fetch('http://localhost:3031/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: etlData.processed_data || etlData.data,
          goal: goal || 'machine learning preparation'
        })
      });

      if (!preprocessResponse.ok) {
        throw new Error('Preprocessing failed');
      }

      const preprocessData = await preprocessResponse.json();

      // Step 3: EDA Service
      setAnalysisStep('Generating visualizations...');
      const edaResponse = await fetch('http://localhost:3035/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: preprocessData.processed_data || preprocessData.data,
          prompt: goal || 'Comprehensive data analysis'
        })
      });

      if (!edaResponse.ok) {
        throw new Error('EDA analysis failed');
      }

      const edaData = await edaResponse.json();

      // Step 4: ML Analysis Service
      setAnalysisStep('Running machine learning analysis...');
      const mlResponse = await fetch('http://localhost:3040/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: preprocessData.processed_data || preprocessData.data,
          goal: goal || 'comprehensive analysis'
        })
      });

      if (!mlResponse.ok) {
        throw new Error('ML analysis failed');
      }

      const mlData = await mlResponse.json();

      // Combine all results
      return {
        etl: etlData,
        preprocessing: preprocessData,
        eda: edaData,
        ml: mlData,
        goal: goal,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      console.error('Analysis pipeline error:', error);
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
        completed++;
      }
      
      setIsUploadComplete(true);
      setIsUploading(false);
      setUploadProgress(0);
      
      // Now start the analysis
      await handleAnalyzeData();
      
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadError(error.message || 'Upload failed. Please try again.');
      setIsUploading(false);
      setUploadProgress(0);
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

    try {
      // Get the uploaded file data from localStorage
      const storedFiles = JSON.parse(localStorage.getItem('walrusFiles') || '[]');
      const uploadedFile = storedFiles.find(file => file.blobId === uploadedBlobId);
      
      if (!uploadedFile) {
        throw new Error('Uploaded file not found');
      }

      // For demo purposes, we'll create sample data if no analysis data exists
      // In a real implementation, you'd read the actual file content
      const analysisData = uploadedFile.analysisData || [
        { name: 'Sample Data', value: 100, category: 'A' },
        { name: 'Demo Entry', value: 200, category: 'B' }
      ];

      const results = await runAnalysisPipeline({ analysisData }, analysisGoal);
      
      // Store analysis results with the file
      const updatedFile = {
        ...uploadedFile,
        analysisResults: results,
        analysisGoal: analysisGoal,
        hasAnalysis: true,
        lastAnalyzed: new Date().toISOString()
      };

      const updatedFiles = storedFiles.map(file => 
        file.blobId === uploadedBlobId ? updatedFile : file
      );

      localStorage.setItem('walrusFiles', JSON.stringify(updatedFiles));
      
      setAnalysisResults(results);
      setAnalysisStep('Analysis complete!');
      
      // Keep modal open to show results instead of auto-closing
      setIsAnalyzing(false);

    } catch (error) {
      setAnalysisError(error.message);
      setIsAnalyzing(false);
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
    <div className="h-screen font-montserrat bg-white text-gray-900 transition-colors duration-300 overflow-hidden flex flex-col">
      {/* Navigation Bar */}
      <motion.nav 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="relative flex justify-center pt-8 pb-4 px-8 flex-shrink-0"
      >
        <div className="flex bg-gray-200 rounded-full p-1 transition-all duration-300 shadow-lg hover:shadow-xl">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full hover:bg-gray-300 text-black font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full bg-gray-600 text-white font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95 shadow-md">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full hover:bg-gray-300 text-black font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              View
            </div>
          </Link>
          <Link href="/subscribe">
            <div className="px-6 py-2 rounded-full hover:bg-gray-300 text-black font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
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
        <h1 className="text-3xl font-bold text-center text-black transform transition-all duration-300">Upload Your Data</h1>
        <p className="text-center text-gray-600 mt-2 transition-opacity duration-200">
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
                ? 'border-solid border-green-500 bg-green-50 scale-[1.02] shadow-lg'
                : isDragging
                ? 'border-dashed border-gray-500 bg-gray-200 scale-[1.05] shadow-2xl animate-pulse'
                : 'border-dashed border-gray-300 hover:border-gray-400 bg-gray-200 shadow-md hover:shadow-xl'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="space-y-4">
              <div className="flex justify-center">
                {selectedFiles.length > 0 ? (
                  <svg className="w-12 h-12 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                ) : (
                  <svg
                    className={`w-12 h-12 text-gray-600 transition-all duration-300 ${isDragging ? 'animate-bounce text-gray-700' : ''}`}
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
                    <p className="text-lg font-medium mb-2 text-green-700">
                      {selectedFiles.map(file => file.name).join(', ')}
                    </p>
                    <p className="text-sm text-black">
                      Ready to upload
                    </p>
                  </motion.div>
                ) : (
                  <div>
                    <p className="text-lg font-medium mb-2 text-black transition-all duration-200">
                      Drop your files here
                    </p>
                    <p className="text-black">
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
                  className="inline-block px-6 py-3 bg-gray-300 text-black rounded-lg font-medium cursor-pointer hover:bg-gray-400 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
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
                    className="px-4 py-2 bg-gray-300 text-black rounded-lg font-medium cursor-pointer hover:bg-gray-400 transition-all duration-200 text-sm transform hover:scale-105 active:scale-95 shadow-sm hover:shadow-md"
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
                    }}
                    className="px-4 py-2 bg-red-100 text-red-700 rounded-lg font-medium hover:bg-red-200 transition-all duration-200 text-sm transform hover:scale-105 active:scale-95 shadow-sm hover:shadow-md"
                  >
                    Remove All
                  </button>
                </motion.div>
              )}
            </div>
          </motion.div>

          {/* File Information */}
          {selectedFiles.length > 0 && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="mt-6 bg-gray-200 p-6 rounded-2xl border border-gray-300 transition-all duration-300 shadow-md hover:shadow-lg"
            >
              <h3 className="font-semibold text-lg mb-4 text-black transform transition-all duration-200">File Details</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-gray-300 rounded-lg transition-all duration-200">
                  <p className="text-sm text-black">Name</p>
                  <p className="font-medium text-black truncate">{selectedFiles[0].name}</p>
                </div>
                <div className="text-center p-4 bg-gray-300 rounded-lg transition-all duration-200">
                  <p className="text-sm text-black">Size</p>
                  <p className="font-medium text-black">{formatFileSize(selectedFiles[0].size)}</p>
                </div>
                <div className="text-center p-4 bg-gray-300 rounded-lg transition-all duration-200">
                  <p className="text-sm text-black">Type</p>
                  <p className="font-medium text-black truncate overflow-hidden whitespace-nowrap max-w-full" style={{display: 'block'}}>{selectedFiles[0].type || 'Unknown'}</p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Analysis Goal Input */}
          {selectedFiles.length > 0 && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="mt-6 bg-gray-200 p-6 rounded-2xl border border-gray-300 transition-all duration-300 shadow-md hover:shadow-lg"
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
              className="mt-6 bg-gray-200 p-6 rounded-2xl border border-gray-300 shadow-lg"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-black">Uploading...</span>
                <span className="text-sm text-black animate-pulse">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-300 rounded-full h-2 overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${uploadProgress}%` }}
                  transition={{ duration: 0.3 }}
                  className="bg-gray-600 h-2 rounded-full animate-pulse"
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
              className="mt-6 bg-red-50 p-6 rounded-2xl border border-red-200 shadow-lg"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <svg className="w-6 h-6 text-red-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-red-700 font-medium">{uploadError}</span>
                </div>
                <button
                  onClick={() => setUploadError('')}
                  className="text-red-500 hover:text-red-700 ml-2"
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
              className="mt-6 bg-green-50 p-4 rounded-2xl border border-green-200 shadow-lg"
            >
              <div className="text-center">
                <div className="flex items-center justify-center">
                  <svg className="w-6 h-6 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-green-700 font-medium">Upload completed successfully!</span>
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

                      {/* EDA Visualizations */}
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

                      {/* AI Insights */}
                      {analysisResults.eda?.analysis?.insights && (
                        <div className="space-y-4">
                          <h3 className="text-xl font-semibold text-gray-800 border-b pb-2">🧠 AI Insights</h3>
                          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                            <div className="space-y-3">
                              {Object.values(analysisResults.eda.analysis.insights.ai_insights || {}).map((category, idx) => (
                                <div key={idx} className="bg-white p-3 rounded border">
                                  <h4 className="font-medium text-blue-900 mb-2">Key Findings #{idx + 1}</h4>
                                  {category.insights && Array.isArray(category.insights) && (
                                    <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                                      {category.insights.slice(0, 3).map((insight, insightIdx) => (
                                        <li key={insightIdx}>{typeof insight === 'string' ? insight : JSON.stringify(insight)}</li>
                                      ))}
                                    </ul>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Machine Learning Results */}
                      {analysisResults.ml?.results?.analyses && (
                        <div className="space-y-6">
                          <h3 className="text-xl font-semibold text-gray-800 border-b pb-2">🤖 Machine Learning Analysis</h3>
                          
                          {/* TEE Attestation */}
                          {analysisResults.ml.results.tee_attestation && (
                            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                              <h4 className="font-medium text-green-900 mb-2 flex items-center gap-2">
                                🔐 TEE Attestation
                                {analysisResults.ml.results.tee_attestation.tee_attested ? (
                                  <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">Verified</span>
                                ) : (
                                  <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">Unavailable</span>
                                )}
                              </h4>
                              <p className="text-sm text-green-700">
                                {analysisResults.ml.results.tee_attestation.tee_attested 
                                  ? "Analysis executed and signed in a Trusted Execution Environment"
                                  : `TEE attestation failed: ${analysisResults.ml.results.tee_attestation.error}`
                                }
                              </p>
                            </div>
                          )}

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
