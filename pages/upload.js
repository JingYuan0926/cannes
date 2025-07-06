import Link from "next/link";
import { useState } from "react";
import { motion } from "framer-motion";
import { uploadFile } from "../utils/writeToWalrus";
import { getUserEncryptionKey, createEncryptedFile, uint8ArrayToBase64 } from "../utils/encryption";
import WalletConnect from '../components/WalletConnect';

export default function Upload() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isUploadComplete, setIsUploadComplete] = useState(false);
  const [uploadedBlobId, setUploadedBlobId] = useState(null);
  const [uploadError, setUploadError] = useState('');

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
              Analyse
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
                    <p className="text-sm text-slate-700">
                      Ready to upload
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
                    }}
                    className="px-4 py-2 bg-rose-100 text-rose-700 rounded-lg font-medium hover:bg-rose-200 transition-all duration-200 text-sm transform hover:scale-105 active:scale-95 shadow-sm hover:shadow-md"
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
              className="mt-6 bg-white/80 backdrop-blur-sm p-6 rounded-2xl border border-blue-200 transition-all duration-300 shadow-md hover:shadow-lg"
            >
              <h3 className="font-semibold text-lg mb-4 text-slate-800 transform transition-all duration-200">File Details</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-blue-100 rounded-lg transition-all duration-200">
                  <p className="text-sm text-slate-700">Name</p>
                  <p className="font-medium text-slate-800 truncate">{selectedFiles[0].name}</p>
                </div>
                <div className="text-center p-4 bg-blue-100 rounded-lg transition-all duration-200">
                  <p className="text-sm text-slate-700">Size</p>
                  <p className="font-medium text-slate-800">{formatFileSize(selectedFiles[0].size)}</p>
                </div>
                <div className="text-center p-4 bg-blue-100 rounded-lg transition-all duration-200">
                  <p className="text-sm text-slate-700">Type</p>
                  <p className="font-medium text-slate-800 truncate overflow-hidden whitespace-nowrap max-w-full" style={{display: 'block'}}>{selectedFiles[0].type || 'Unknown'}</p>
                </div>
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
          
          {/* Upload Button */}
          <motion.div 
            variants={itemVariants}
            className="mt-8 text-center"
          >
            <button
              onClick={handleUpload}
              disabled={selectedFiles.length === 0 || isUploading}
              className={`px-8 py-3 rounded-lg font-medium transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl ${
                selectedFiles.length > 0 && !isUploading
                  ? 'bg-blue-600 text-white hover:bg-blue-700 hover:-translate-y-1'
                  : 'bg-blue-200 text-slate-500 cursor-not-allowed scale-100'
              }`}
            >
              {isUploading ? (
                <div className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Uploading...
                </div>
              ) : (
                'Upload Files'
              )}
            </button>
            
            {/* Encryption Notice */}
            {selectedFiles.length > 0 && !isUploading && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="mt-4 flex items-center justify-center text-sm text-slate-600"
              >
                <svg className="w-4 h-4 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                <span>Your files will be encrypted before upload for security</span>
              </motion.div>
            )}
          </motion.div>
        </div>
      </motion.main>
    </div>
  );
}
