import Link from "next/link";
import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { readFW, readContentWithType, parseCSV, detectFileType, downloadFile } from "../utils/readFromWalrus";

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
        setFiles(storedFiles);
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
      
      // Detect file type and add additional metadata
      let fileType = detectFileType(result.content, result.contentType, result.blobId);
      
      const contentData = {
        ...result,
        fileType,
      };

      // Parse CSV data if it's a CSV file
      if (fileType.type === 'csv' && result.isText) {
        try {
          contentData.csvData = parseCSV(result.content);
        } catch (csvError) {
          console.warn('Failed to parse CSV:', csvError);
          contentData.csvData = null;
        }
      }

      // Try to parse JSON if it's JSON content
      if (fileType.type === 'json' && result.isText) {
        try {
          contentData.jsonData = JSON.parse(result.content);
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
      await downloadFile(file.blobId, file.name);
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
    <div className="h-screen font-montserrat bg-white text-gray-900 transition-colors duration-300 overflow-hidden flex flex-col">
      {/* Navigation Bar */}
      <motion.nav 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="flex justify-center pt-8 pb-4 flex-shrink-0"
      >
        <div className="flex bg-gray-200 rounded-full p-1 transition-all duration-300 shadow-lg hover:shadow-xl">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full hover:bg-gray-300 text-black font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full hover:bg-gray-300 text-black font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full bg-gray-600 text-white font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95 shadow-md">
              View
            </div>
          </Link>
        </div>
      </motion.nav>

      {/* Page Header */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut", delay: 0.1 }}
        className="px-8 py-4 flex-shrink-0"
      >
        <h1 className="text-3xl font-bold text-center text-black transform transition-all duration-300">View Your Data</h1>
        <p className="text-center text-gray-600 mt-2 transition-opacity duration-200">
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
            className="bg-gray-200 rounded-2xl p-6 border border-gray-300 transition-all duration-300 shadow-md"
          >
            <div>
              <h3 className="text-2xl font-bold text-black">{files.length}</h3>
              <p className="text-black text-sm">Total Files</p>
            </div>
          </motion.div>
          
          <motion.div 
            variants={statsVariants}
            className="bg-gray-200 rounded-2xl p-6 border border-gray-300 transition-all duration-300 shadow-md"
          >
            <div>
              <h3 className="text-2xl font-bold text-green-600">
                {files.filter(file => file.isActive).length}
              </h3>
              <p className="text-black text-sm">Active Files</p>
            </div>
          </motion.div>
          
          <motion.div 
            variants={statsVariants}
            className="bg-gray-200 rounded-2xl p-6 border border-gray-300 transition-all duration-300 shadow-md"
          >
            <div>
              <h3 className="text-2xl font-bold text-red-600">
                {files.filter(file => !file.isActive).length}
              </h3>
              <p className="text-black text-sm">Inactive Files</p>
            </div>
          </motion.div>
          
          <motion.div 
            variants={statsVariants}
            className="bg-gray-200 rounded-2xl p-6 border border-gray-300 transition-all duration-300 shadow-md"
          >
            <div>
              <h3 className="text-2xl font-bold text-black">
                {files.length > 0 
                  ? (files.reduce((total, file) => total + (file.originalSize || 0), 0) / (1024 * 1024)).toFixed(1) + ' MB'
                  : '0 MB'
                }
              </h3>
              <p className="text-black text-sm">Total Size</p>
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
              <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-gray-200 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent transition-all duration-300 hover:shadow-md focus:shadow-lg text-black placeholder-gray-600"
              />
            </div>
          </div>
          
          {/* Custom Dropdown */}
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex items-center justify-between w-full sm:w-40 px-4 py-3 bg-gray-200 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent transition-all duration-300 hover:shadow-md focus:shadow-lg transform hover:scale-105 focus:scale-105 text-black"
            >
              <div className="flex items-center">
                <span className="text-sm font-medium">
                  {filterOptions.find(option => option.value === filterStatus)?.label}
                </span>
              </div>
              <svg 
                className={`w-4 h-4 text-gray-600 transition-transform duration-200 mr-1 ${isDropdownOpen ? 'rotate-180' : ''}`}
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
                  className="absolute top-full left-0 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg overflow-hidden z-50"
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
                      className={`w-full px-4 py-3 text-left hover:bg-gray-50 transition-all duration-200 flex items-center justify-between ${
                        filterStatus === option.value 
                          ? 'bg-gray-100 border-l-4 border-gray-500' 
                          : 'border-l-4 border-transparent'
                      }`}
                    >
                      <span className="text-sm font-medium text-gray-900">{option.label}</span>
                      {filterStatus === option.value && (
                        <svg className="w-4 h-4 text-gray-600 ml-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
          className="bg-gray-200 rounded-2xl border border-gray-300 overflow-hidden transition-all duration-300 shadow-lg hover:shadow-xl"
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[800px]">
              <thead className="bg-gray-300 transition-colors duration-300">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200 w-1/3">File</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200 w-20">Size</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200 w-24">Modified</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200 w-20">Status</th>
                  <th className="px-6 py-4 text-left text-xs font-medium text-black uppercase tracking-wider hover:text-gray-800 transition-colors duration-200 w-32">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-300">
                {filteredFiles.map((file, index) => (
                  <motion.tr 
                    key={file.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    className="hover:bg-gray-300 transition-all duration-200 group hover:shadow-md"
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-black group-hover:text-gray-800 transition-colors duration-200">
                        {file.name}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-black group-hover:text-gray-800 transition-colors duration-200">
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
                            ? 'bg-green-100 text-green-800 hover:bg-green-200 hover:shadow-md'
                            : 'bg-red-100 text-red-800 hover:bg-red-200 hover:shadow-md'
                        }`}
                      >
                        {file.isActive ? 'Active' : 'Inactive'}
                      </button>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleDownloadFile(file)}
                          className="text-black hover:text-gray-800 transition-all duration-200 font-medium transform hover:scale-105 active:scale-95"
                        >
                          Download
                        </button>
                        <button
                          onClick={() => handleDeleteFile(file.id)}
                          className="text-red-600 hover:text-red-800 transition-all duration-200 font-medium transform hover:scale-105 active:scale-95"
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
              <p className="text-black text-lg mb-2">
                {files.length === 0 ? "No files uploaded yet" : "No files found matching your criteria"}
              </p>
              {files.length === 0 && (
                <div className="mt-4">
                  <Link href="/upload">
                    <button className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-all duration-200">
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
            className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg"
          >
            <div className="flex justify-between items-start">
              <p className="text-red-700 text-sm">{contentError}</p>
              <button
                onClick={() => setContentError('')}
                className="text-red-500 hover:text-red-700 ml-2"
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
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
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
              <div className="p-6 border-b border-gray-200 bg-gray-50">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-lg font-semibold text-black">{viewingFile.name}</h3>
                    <p className="text-sm text-gray-600 mt-1">
                      Blob ID: {viewingFile.blobId}
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      setViewingFile(null);
                      setFileContent(null);
                      setContentError('');
                    }}
                    className="text-gray-400 hover:text-gray-600 transition-all duration-200 transform hover:scale-110 active:scale-90"
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
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-600 mx-auto"></div>
                    <p className="text-gray-600 mt-2">Loading file content...</p>
                  </div>
                ) : fileContent ? (
                  <div>
                    {/* File Type Info */}
                    <div className="mb-4 p-3 bg-gray-200 rounded-lg">
                      <p className="text-sm text-black">
                        <strong>Type:</strong> {fileContent.fileType?.displayName || 'Unknown'} • 
                        <strong> Size:</strong> {Math.round(fileContent.bytes / 1024)} KB • 
                        <strong> Content-Type:</strong> {fileContent.contentType}
                      </p>
                    </div>

                    {/* Render Content Based on Type */}
                    {fileContent.fileType?.type === 'csv' && fileContent.csvData ? (
                      <div>
                        <h4 className="font-semibold mb-2 text-black">CSV Data ({fileContent.csvData.length} rows):</h4>
                        <div className="overflow-x-auto max-h-96 border rounded-lg shadow-md">
                          <table className="min-w-full text-sm">
                            <thead className="bg-gray-200">
                              <tr>
                                {Object.keys(fileContent.csvData[0] || {}).map((header) => (
                                  <th key={header} className="px-3 py-2 text-left font-medium border-b text-black">
                                    {header}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {fileContent.csvData.slice(0, 50).map((row, index) => (
                                <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                  {Object.values(row).map((cell, cellIndex) => (
                                    <td key={cellIndex} className="px-3 py-2 border-b text-black">
                                      {cell}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {fileContent.csvData.length > 50 && (
                            <p className="text-xs text-gray-600 p-2">
                              Showing first 50 rows of {fileContent.csvData.length} total rows
                            </p>
                          )}
                        </div>
                      </div>
                    ) : fileContent.fileType?.type === 'json' && fileContent.jsonData ? (
                      <div>
                        <h4 className="font-semibold mb-2 text-black">JSON Data:</h4>
                        <pre className="text-sm bg-gray-200 p-4 rounded-lg border overflow-auto max-h-96 shadow-md">
                          {JSON.stringify(fileContent.jsonData, null, 2)}
                        </pre>
                      </div>
                    ) : fileContent.isText ? (
                      <div>
                        <h4 className="font-semibold mb-2 text-black">Text Content:</h4>
                        <pre className="text-sm bg-gray-200 p-4 rounded-lg border overflow-auto max-h-96 whitespace-pre-wrap shadow-md">
                          {fileContent.content}
                        </pre>
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <p className="text-gray-600 mb-4">Binary file - cannot preview</p>
                        <button
                          onClick={() => handleDownloadFile(viewingFile)}
                          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                        >
                          Download File
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-gray-600">Failed to load file content</p>
                  </div>
                )}
              </div>

              {/* Modal Footer */}
              {fileContent && (
                <div className="p-6 border-t border-gray-200 bg-gray-50">
                  <div className="flex justify-between items-center">
                    <button
                      onClick={() => navigator.clipboard.writeText(viewingFile.blobId)}
                      className="px-4 py-2 bg-gray-200 text-black rounded-lg hover:bg-gray-300 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                    >
                      Copy Blob ID
                    </button>
                    <button
                      onClick={() => handleDownloadFile(viewingFile)}
                      className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
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
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
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
              <div className="p-6 border-b border-gray-200 bg-gray-50">
                <h3 className="text-lg font-semibold text-black">Delete File</h3>
              </div>

              {/* Modal Content */}
              <div className="p-6 bg-white">
                <p className="text-gray-700 mb-4">
                  Are you sure you want to delete <strong>{deleteConfirmFile.name}</strong> from your local list?
                </p>
                <p className="text-sm text-gray-600">
                  This won't delete it from Walrus storage.
                </p>
              </div>

              {/* Modal Footer */}
              <div className="p-6 border-t border-gray-200 bg-gray-50">
                <div className="flex justify-end space-x-3">
                  <button
                    onClick={cancelDeleteFile}
                    className="px-4 py-2 bg-gray-200 text-black rounded-lg hover:bg-gray-300 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={confirmDeleteFile}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
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
