import Link from "next/link";
import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { readFW, readContentWithType, parseCSV, detectFileType, downloadFile } from "../utils/readFromWalrus";
import { getUserEncryptionKey, decryptData, base64ToUint8Array, base64ToArrayBuffer } from "../utils/encryption";
import WalletConnect from '../components/WalletConnect';

export default function View() {
  const [files, setFiles] = useState([]);

  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [viewingFile, setViewingFile] = useState(null);
  const [fileContent, setFileContent] = useState(null);
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [contentError, setContentError] = useState('');
  const [deleteConfirmFile, setDeleteConfirmFile] = useState(null);
  const dropdownRef = useRef(null);

  const filterOptions = [
    { value: "all", label: "All Files" },
    { value: "active", label: "Active Only" },
    { value: "inactive", label: "Inactive Only" }
  ];

  // Load files from localStorage on mount
  useEffect(() => {
    const loadFiles = () => {
      try {
        const storedFiles = JSON.parse(localStorage.getItem('walrusFiles') || '[]');
        // Remove duplicates by name, keeping the newest (by timestamp or id)
        const uniqueFilesMap = {};
        for (const file of storedFiles) {
          // If not seen or this one is newer, keep it
          if (!uniqueFilesMap[file.name] || new Date(file.timestamp) > new Date(uniqueFilesMap[file.name].timestamp)) {
            uniqueFilesMap[file.name] = file;
          }
        }
        const uniqueFiles = Object.values(uniqueFilesMap);
        setFiles(uniqueFiles);
        // Optionally, update localStorage to keep it clean
        localStorage.setItem('walrusFiles', JSON.stringify(uniqueFiles));
      } catch (error) {
        console.error('Failed to load files from localStorage:', error);
        setFiles([]);
      }
    };
    
    loadFiles();
    
    // Listen for storage changes (when files are uploaded)
    const handleStorageChange = () => {
      loadFiles();
    };
    
    window.addEventListener('storage', handleStorageChange);
    
    // Also listen for custom events from the upload page
    window.addEventListener('walrusFileUploaded', handleStorageChange);
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('walrusFileUploaded', handleStorageChange);
    };
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const toggleFileStatus = (id) => {
    const updatedFiles = files.map(file => 
      file.id === id ? { ...file, isActive: !file.isActive } : file
    );
    setFiles(updatedFiles);
    // Update localStorage
    localStorage.setItem('walrusFiles', JSON.stringify(updatedFiles));
  };

  const handleViewFile = async (file) => {
    if (!file.blobId) {
      setContentError('No Blob ID available for this file');
      return;
    }

    setViewingFile(file);
    setIsLoadingContent(true);
    setContentError('');
    setFileContent(null);

    try {
      // Use enhanced read function with content type detection
      const result = await readContentWithType(file.blobId);
      
      let contentData = { ...result };
      
      // Check if file is encrypted and decrypt if needed
      if (file.isEncrypted && file.encryptionIV) {
        try {
          // Get user's encryption key
          const encryptionKey = await getUserEncryptionKey();
          
          // Convert IV from base64 to Uint8Array
          const iv = base64ToUint8Array(file.encryptionIV);
          
          // Convert encrypted content to ArrayBuffer
          let encryptedBuffer;
          if (result.isBinary) {
            // Content is already base64 encoded for binary files
            encryptedBuffer = base64ToArrayBuffer(result.content);
          } else {
            // For text content, convert to ArrayBuffer
            encryptedBuffer = new TextEncoder().encode(result.content);
          }
          
          // Decrypt the content
          const decryptedBuffer = await decryptData(encryptedBuffer, iv, encryptionKey);
          
          // Create a blob from decrypted data to determine its type
          const decryptedBlob = new Blob([decryptedBuffer]);
          
          // Use original file metadata for type detection
          const originalMetadata = file.encryptionMetadata || {};
          const originalType = originalMetadata.originalType || file.type;
          
          // Detect if decrypted content is text or binary
          const isTextType = originalType && (
            originalType.startsWith('text/') || 
            originalType.includes('json') || 
            originalType.includes('xml') || 
            originalType.includes('csv')
          );
          
          if (isTextType) {
            // Convert decrypted buffer to text
            const decryptedText = new TextDecoder().decode(decryptedBuffer);
            contentData = {
              ...contentData,
              content: decryptedText,
              contentType: originalType,
              isText: true,
              isBinary: false,
              length: decryptedText.length,
              bytes: decryptedBuffer.byteLength,
              isDecrypted: true,
            };
          } else {
            // Keep as binary, convert to base64
            const decryptedBase64 = btoa(String.fromCharCode(...new Uint8Array(decryptedBuffer)));
            contentData = {
              ...contentData,
              content: decryptedBase64,
              contentType: originalType,
              isText: false,
              isBinary: true,
              length: decryptedBuffer.byteLength,
              bytes: decryptedBuffer.byteLength,
              isDecrypted: true,
            };
          }
        } catch (decryptError) {
          console.error('Decryption failed:', decryptError);
          setContentError(`Failed to decrypt file: ${decryptError.message}`);
          return;
        }
      }
      
      // Detect file type and add additional metadata
      let fileType = detectFileType(contentData.content, contentData.contentType, result.blobId);
      
      contentData.fileType = fileType;

      // Parse CSV data if it's a CSV file
      if (fileType.type === 'csv' && contentData.isText) {
        try {
          contentData.csvData = parseCSV(contentData.content);
        } catch (csvError) {
          console.warn('Failed to parse CSV:', csvError);
          contentData.csvData = null;
        }
      }

      // Try to parse JSON if it's JSON content
      if (fileType.type === 'json' && contentData.isText) {
        try {
          contentData.jsonData = JSON.parse(contentData.content);
        } catch (jsonError) {
          console.warn('Failed to parse JSON:', jsonError);
          contentData.jsonData = null;
        }
      }

      setFileContent(contentData);
    } catch (error) {
      setContentError(`Failed to load file content: ${error.message}`);
    } finally {
      setIsLoadingContent(false);
    }
  };

  const handleDeleteFile = (id) => {
    const fileToDelete = files.find(file => file.id === id);
    setDeleteConfirmFile(fileToDelete);
  };

  const confirmDeleteFile = () => {
    if (deleteConfirmFile) {
      const updatedFiles = files.filter(file => file.id !== deleteConfirmFile.id);
      setFiles(updatedFiles);
      localStorage.setItem('walrusFiles', JSON.stringify(updatedFiles));
      
      // Close file content view if this file was being viewed
      if (viewingFile && viewingFile.id === deleteConfirmFile.id) {
        setViewingFile(null);
        setFileContent(null);
      }
      
      setDeleteConfirmFile(null);
    }
  };

  const cancelDeleteFile = () => {
    setDeleteConfirmFile(null);
  };

  const handleDownloadFile = async (file) => {
    if (!file.blobId) {
      setContentError('No Blob ID available for download');
      return;
    }

    try {
      // If file is encrypted, we need to decrypt it first
      if (file.isEncrypted && file.encryptionIV) {
        // Read the encrypted content
        const result = await readContentWithType(file.blobId);
        
        // Get user's encryption key
        const encryptionKey = await getUserEncryptionKey();
        
        // Convert IV from base64 to Uint8Array
        const iv = base64ToUint8Array(file.encryptionIV);
        
        // Convert encrypted content to ArrayBuffer
        let encryptedBuffer;
        if (result.isBinary) {
          encryptedBuffer = base64ToArrayBuffer(result.content);
        } else {
          encryptedBuffer = new TextEncoder().encode(result.content);
        }
        
        // Decrypt the content
        const decryptedBuffer = await decryptData(encryptedBuffer, iv, encryptionKey);
        
        // Create blob and download
        const originalMetadata = file.encryptionMetadata || {};
        const originalType = originalMetadata.originalType || file.type;
        const originalName = originalMetadata.originalName || file.name;
        
        const blob = new Blob([decryptedBuffer], { type: originalType });
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = originalName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        console.log(`Decrypted file downloaded: ${originalName}`);
      } else {
        // Use regular download for non-encrypted files
        await downloadFile(file.blobId, file.name);
      }
    } catch (error) {
      setContentError(`Failed to download file: ${error.message}`);
    }
  };

  const filteredFiles = files.filter(file => {
    const matchesSearch = file.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterStatus === "all" || 
                         (filterStatus === "active" && file.isActive) ||
                         (filterStatus === "inactive" && !file.isActive);
    return matchesSearch && matchesFilter;
  });

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

  const statsVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: {
        duration: 0.5,
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
            <div className="px-6 py-2 rounded-full hover:bg-blue-100 text-slate-700 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full bg-blue-600 text-white font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95 shadow-md">
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
        <h1 className="text-3xl font-bold text-center text-slate-800 transform transition-all duration-300">View Your Data</h1>
        <p className="text-center text-slate-600 mt-2 transition-opacity duration-200">
          Manage and explore your uploaded datasets
        </p>
      </motion.div>

      {/* Main Content */}
      <motion.main 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="flex-1 max-w-7xl mx-auto px-8 py-8 w-full min-h-0 overflow-y-auto"
      >
        
        {/* Summary Stats */}
        <motion.div 
          variants={containerVariants}
          className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8"
        >
          <motion.div 
            variants={statsVariants}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-blue-200 transition-all duration-300 shadow-md hover:shadow-lg"
          >
            <div>
              <h3 className="text-2xl font-bold text-slate-800">{files.length}</h3>
              <p className="text-slate-600 text-sm">Total Files</p>
            </div>
          </motion.div>
          
          <motion.div 
            variants={statsVariants}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-blue-200 transition-all duration-300 shadow-md hover:shadow-lg"
          >
            <div>
              <h3 className="text-2xl font-bold text-emerald-600">
                {files.filter(file => file.isActive).length}
              </h3>
              <p className="text-slate-600 text-sm">Active Files</p>
            </div>
          </motion.div>
          
          <motion.div 
            variants={statsVariants}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-blue-200 transition-all duration-300 shadow-md hover:shadow-lg"
          >
            <div>
              <h3 className="text-2xl font-bold text-rose-600">
                {files.filter(file => !file.isActive).length}
              </h3>
              <p className="text-slate-600 text-sm">Inactive Files</p>
            </div>
          </motion.div>
          
          <motion.div 
            variants={statsVariants}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-blue-200 transition-all duration-300 shadow-md hover:shadow-lg"
          >
            <div>
              <h3 className="text-2xl font-bold text-slate-800">
                {files.length > 0 
                  ? (files.reduce((total, file) => total + (file.originalSize || 0), 0) / (1024 * 1024)).toFixed(1) + ' MB'
                  : '0 MB'
                }
              </h3>
              <p className="text-slate-600 text-sm">Total Size</p>
            </div>
          </motion.div>
        </motion.div>

        {/* Search and Filter */}
        <motion.div 
          variants={itemVariants}
          className="flex flex-col sm:flex-row gap-4 mb-6"
        >
          <div className="flex-1">
            <div className="relative group">
              <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-white/80 backdrop-blur-sm border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 hover:shadow-md focus:shadow-lg text-slate-800 placeholder-slate-500"
              />
            </div>
          </div>
          
          {/* Custom Dropdown */}
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex items-center justify-between w-full sm:w-40 px-4 py-3 bg-white/80 backdrop-blur-sm border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 hover:shadow-md focus:shadow-lg transform hover:scale-105 focus:scale-105 text-slate-800"
            >
              <div className="flex items-center">
                <span className="text-sm font-medium">
                  {filterOptions.find(option => option.value === filterStatus)?.label}
                </span>
              </div>
              <svg 
                className={`w-4 h-4 text-slate-500 transition-transform duration-200 mr-1 ${isDropdownOpen ? 'rotate-180' : ''}`}
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            <AnimatePresence>
              {isDropdownOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  transition={{ duration: 0.15, ease: "easeOut" }}
                  className="absolute top-full left-0 w-full mt-1 bg-white/95 backdrop-blur-sm border border-blue-200 rounded-lg shadow-lg overflow-hidden z-50"
                >
                  {filterOptions.map((option, index) => (
                    <motion.button
                      key={option.value}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.1, delay: index * 0.05 }}
                      onClick={() => {
                        setFilterStatus(option.value);
                        setIsDropdownOpen(false);
                      }}
                      className={`w-full px-4 py-3 text-left hover:bg-blue-50 transition-all duration-200 flex items-center justify-between ${
                        filterStatus === option.value 
                          ? 'bg-blue-100 border-l-4 border-blue-500' 
                          : 'border-l-4 border-transparent'
                      }`}
                    >
                      <span className="text-sm font-medium text-slate-800">{option.label}</span>
                      {filterStatus === option.value && (
                        <svg className="w-4 h-4 text-blue-600 ml-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </motion.button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
          
        {/* Files Table */}
        <motion.div 
          variants={itemVariants}
          className="bg-white/80 backdrop-blur-sm rounded-2xl border border-blue-200 overflow-hidden transition-all duration-300 shadow-lg hover:shadow-xl"
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[800px]">
              <thead className="bg-blue-100/80 backdrop-blur-sm transition-colors duration-300">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider hover:text-slate-900 transition-colors duration-200 w-1/3">File</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider hover:text-slate-900 transition-colors duration-200 w-20">Size</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider hover:text-slate-900 transition-colors duration-200 w-24">Modified</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider hover:text-slate-900 transition-colors duration-200 w-20">Status</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider hover:text-slate-900 transition-colors duration-200 w-20">Security</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider hover:text-slate-900 transition-colors duration-200 w-32">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-blue-200">
                {filteredFiles.map((file, index) => (
                  <motion.tr 
                    key={file.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    className="hover:bg-blue-50/50 transition-all duration-200 group hover:shadow-md"
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-slate-800 group-hover:text-slate-900 transition-colors duration-200">
                        {file.name}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700 group-hover:text-slate-900 transition-colors duration-200">
                      {file.size}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {new Date(file.timestamp).toLocaleDateString('en-GB', {
                        day: '2-digit',
                        month: '2-digit', 
                        year: 'numeric'
                      })}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button
                        onClick={() => toggleFileStatus(file.id)}
                        className={`inline-flex px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 transform hover:scale-105 active:scale-95 ${
                          file.isActive
                            ? 'bg-emerald-100 text-emerald-800 hover:bg-emerald-200 hover:shadow-md'
                            : 'bg-rose-100 text-rose-800 hover:bg-rose-200 hover:shadow-md'
                        }`}
                      >
                        {file.isActive ? 'Active' : 'Inactive'}
                      </button>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className={`inline-flex px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 ${
                        file.isEncrypted
                          ? 'bg-blue-100 text-blue-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}>
                        {file.isEncrypted ? (
                          <div className="flex items-center">
                            <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                            </svg>
                            Encrypted
                          </div>
                        ) : (
                          <div className="flex items-center">
                            <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z" />
                            </svg>
                            Plain
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex space-x-4">
                        <button
                          onClick={() => handleViewFile(file)}
                          className="text-blue-600 hover:text-blue-800 transition-all duration-200 font-medium transform hover:scale-105 active:scale-95"
                        >
                          View
                        </button>
                        <button
                          onClick={() => handleDownloadFile(file)}
                          className="text-indigo-600 hover:text-indigo-800 transition-all duration-200 font-medium transform hover:scale-105 active:scale-95"
                        >
                          Download
                        </button>
                        <button
                          onClick={() => handleDeleteFile(file.id)}
                          className="text-rose-600 hover:text-rose-800 transition-all duration-200 font-medium transform hover:scale-105 active:scale-95"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {filteredFiles.length === 0 && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="text-center py-12"
            >
              <p className="text-slate-700 text-lg mb-2">
                {files.length === 0 ? "No files uploaded yet" : "No files found matching your criteria"}
              </p>
              {files.length === 0 && (
                <div className="mt-4">
                  <Link href="/upload">
                    <button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-200">
                      Upload Your First File
                    </button>
                  </Link>
                </div>
              )}
            </motion.div>
          )}
        </motion.div>

        {/* Error Display */}
        {contentError && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="mt-6 p-4 bg-rose-50 border border-rose-200 rounded-lg"
          >
            <div className="flex justify-between items-start">
              <p className="text-rose-700 text-sm">{contentError}</p>
              <button
                onClick={() => setContentError('')}
                className="text-rose-500 hover:text-rose-700 ml-2"
              >
                ×
              </button>
            </div>
          </motion.div>
        )}
      </motion.main>

      {/* File Content Modal */}
      <AnimatePresence>
        {viewingFile && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={() => {
              setViewingFile(null);
              setFileContent(null);
              setContentError('');
            }}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="p-6 border-b border-blue-200 bg-blue-50/80 backdrop-blur-sm">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-lg font-semibold text-slate-800">{viewingFile.name}</h3>
                    <p className="text-sm text-slate-600 mt-1">
                      Blob ID: <a 
                        href={`https://walruscan.com/testnet/blob/${viewingFile.blobId}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline text-blue-600 hover:text-blue-800 transition-colors duration-200 inline-flex items-center gap-1"
                      >
                        {viewingFile.blobId}
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                      </a>
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      setViewingFile(null);
                      setFileContent(null);
                      setContentError('');
                    }}
                    className="text-slate-400 hover:text-slate-600 transition-all duration-200 transform hover:scale-110 active:scale-90"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* Modal Content */}
              <div className="p-6 overflow-y-auto max-h-[70vh] bg-white">
                {isLoadingContent ? (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                    <p className="text-slate-600 mt-2">Loading file content...</p>
                  </div>
                ) : fileContent ? (
                  <div>
                    {/* File Type Info */}
                    <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                      <p className="text-sm text-slate-700">
                        <strong>Type:</strong> {fileContent.fileType?.displayName || 'Unknown'} • 
                        <strong> Size:</strong> {Math.round(fileContent.bytes / 1024)} KB • 
                        <strong> Content-Type:</strong> {fileContent.contentType}
                        {fileContent.isDecrypted && (
                          <>
                            {' • '}
                            <span className="inline-flex items-center text-emerald-700">
                              <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                              <strong>Decrypted</strong>
                            </span>
                          </>
                        )}
                      </p>
                    </div>

                    {/* Render Content Based on Type */}
                    {fileContent.fileType?.type === 'csv' && fileContent.csvData ? (
                      <div>
                        <h4 className="font-semibold mb-2 text-slate-800">CSV Data ({fileContent.csvData.length} rows):</h4>
                        <div className="overflow-x-auto max-h-96 border border-blue-200 rounded-lg shadow-md">
                          <table className="min-w-full text-sm">
                            <thead className="bg-blue-100">
                              <tr>
                                {Object.keys(fileContent.csvData[0] || {}).map((header) => (
                                  <th key={header} className="px-3 py-2 text-left font-medium border-b border-blue-200 text-slate-800">
                                    {header}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {fileContent.csvData.slice(0, 50).map((row, index) => (
                                <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-blue-50'}>
                                  {Object.values(row).map((cell, cellIndex) => (
                                    <td key={cellIndex} className="px-3 py-2 border-b border-blue-200 text-slate-700">
                                      {cell}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {fileContent.csvData.length > 50 && (
                            <p className="text-xs text-slate-600 p-2">
                              Showing first 50 rows of {fileContent.csvData.length} total rows
                            </p>
                          )}
                        </div>
                      </div>
                    ) : fileContent.fileType?.type === 'json' && fileContent.jsonData ? (
                      <div>
                        <h4 className="font-semibold mb-2 text-slate-800">JSON Data:</h4>
                        <pre className="text-sm bg-blue-50 p-4 rounded-lg border border-blue-200 overflow-auto max-h-96 shadow-md">
                          {JSON.stringify(fileContent.jsonData, null, 2)}
                        </pre>
                      </div>
                    ) : fileContent.isText ? (
                      <div>
                        <h4 className="font-semibold mb-2 text-slate-800">Text Content:</h4>
                        <pre className="text-sm bg-blue-50 p-4 rounded-lg border border-blue-200 overflow-auto max-h-96 whitespace-pre-wrap shadow-md">
                          {fileContent.content}
                        </pre>
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <p className="text-slate-600 mb-4">Binary file - cannot preview</p>
                        <button
                          onClick={() => handleDownloadFile(viewingFile)}
                          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                        >
                          Download File
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-slate-600">Failed to load file content</p>
                  </div>
                )}
              </div>

              {/* Modal Footer */}
              {fileContent && (
                <div className="p-6 border-t border-blue-200 bg-blue-50/80 backdrop-blur-sm">
                  <div className="flex justify-between items-center">
                    <button
                      onClick={() => navigator.clipboard.writeText(viewingFile.blobId)}
                      className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                    >
                      Copy Blob ID
                    </button>
                    <button
                      onClick={() => handleDownloadFile(viewingFile)}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                    >
                      Download File
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Delete Confirmation Modal */}
      <AnimatePresence>
        {deleteConfirmFile && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={cancelDeleteFile}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-lg max-w-md w-full overflow-hidden shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="p-6 border-b border-blue-200 bg-blue-50/80 backdrop-blur-sm">
                <h3 className="text-lg font-semibold text-slate-800">Delete File</h3>
              </div>

              {/* Modal Content */}
              <div className="p-6 bg-white">
                <p className="text-slate-700 mb-4">
                  Are you sure you want to delete <strong>{deleteConfirmFile.name}</strong> from your local list?
                </p>
                <p className="text-sm text-slate-600">
                  This won't delete it from Walrus storage.
                </p>
              </div>

              {/* Modal Footer */}
              <div className="p-6 border-t border-blue-200 bg-blue-50/80 backdrop-blur-sm">
                <div className="flex justify-end space-x-3">
                  <button
                    onClick={cancelDeleteFile}
                    className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={confirmDeleteFile}
                    className="px-4 py-2 bg-rose-600 text-white rounded-lg hover:bg-rose-700 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
