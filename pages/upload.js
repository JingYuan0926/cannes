import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import { useState, useEffect } from "react";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export default function Upload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  // Dark mode persistence
  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedDarkMode !== null) {
      setDarkMode(JSON.parse(savedDarkMode));
    } else {
      setDarkMode(prefersDark);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
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
    const file = event.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      setIsUploading(true);
      setUploadProgress(0);
      
      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setIsUploading(false);
            // Show success message or redirect
            setTimeout(() => {
              alert('File uploaded successfully!');
            }, 500);
            return 100;
          }
          return prev + 10;
        });
      }, 200);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className={`${geistSans.className} ${geistMono.className} min-h-screen font-[family-name:var(--font-geist-sans)] bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-300`}>
      {/* Navigation Bar */}
      <nav className="flex justify-center pt-8 pb-4">
        <div className="flex bg-gray-100 dark:bg-gray-800 rounded-full p-1 transition-colors duration-300">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 font-medium text-sm transition-all duration-300 cursor-pointer">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 font-medium text-sm transition-all duration-300 cursor-pointer">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 font-medium text-sm transition-all duration-300 cursor-pointer">
              View
            </div>
          </Link>
        </div>
      </nav>

      {/* Page Header */}
      <div className="px-8 py-4">
        <h1 className="text-3xl font-bold text-center">Upload Your Data</h1>
        <p className="text-center text-gray-600 dark:text-gray-400 mt-2">
          Upload your datasets to start analyzing and discovering insights
        </p>
      </div>

      {/* Main Content */}
      <main className="flex flex-col items-center justify-center min-h-[calc(100vh-200px)] px-8">
        <div className="max-w-2xl w-full">
          
          {/* File Upload Area */}
          <div
            className={`border-2 rounded-2xl p-12 text-center transition-all duration-300 ${
              selectedFile
                ? 'border-solid border-green-500 dark:border-green-400 bg-green-50 dark:bg-green-950'
                : isDragging
                ? 'border-dashed border-blue-500 dark:border-blue-400 bg-blue-50 dark:bg-blue-950'
                : 'border-dashed border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500 bg-gray-50 dark:bg-gray-800'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="space-y-4">
              <div className="flex justify-center">
                {selectedFile ? (
                  <svg className="w-12 h-12 text-green-500 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                ) : (
                  <svg
                    className="w-12 h-12 text-gray-400 dark:text-gray-500"
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
                {selectedFile ? (
                  <div>
                    <p className="text-lg font-medium mb-2 text-green-700 dark:text-green-300">
                      ✓ {selectedFile.name}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Ready to upload
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className="text-lg font-medium mb-2">
                      Drop your file here
                    </p>
                    <p className="text-gray-600 dark:text-gray-400">
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
              />
              
              {!selectedFile && (
                <label
                  htmlFor="fileInput"
                  className="inline-block px-6 py-3 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors duration-200"
                >
                  Select File
                </label>
              )}

              {selectedFile && (
                <div className="flex gap-2 justify-center">
                  <label
                    htmlFor="fileInput"
                    className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors duration-200 text-sm"
                  >
                    Change File
                  </label>
                  <button
                    onClick={() => setSelectedFile(null)}
                    className="px-4 py-2 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 rounded-lg font-medium hover:bg-red-200 dark:hover:bg-red-800 transition-colors duration-200 text-sm"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Supported file types */}
          <div className="mt-4 text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Supported formats: CSV, Excel (.xlsx, .xls), JSON, TXT
            </p>
          </div>
          
          {/* File Information */}
          {selectedFile && (
            <div className="mt-6 bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-200 dark:border-gray-700 transition-colors duration-300">
              <h3 className="font-semibold text-lg mb-4 text-gray-900 dark:text-gray-100">File Details</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Name</p>
                  <p className="font-medium text-gray-900 dark:text-gray-100 truncate">{selectedFile.name}</p>
                </div>
                <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Size</p>
                  <p className="font-medium text-gray-900 dark:text-gray-100">{formatFileSize(selectedFile.size)}</p>
                </div>
                <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Type</p>
                  <p className="font-medium text-gray-900 dark:text-gray-100">{selectedFile.type || 'Unknown'}</p>
                </div>
              </div>
            </div>
          )}

          {/* Upload Progress */}
          {isUploading && (
            <div className="mt-6 bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Uploading...</span>
                <span className="text-sm text-gray-500 dark:text-gray-400">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-600 dark:bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          )}
          
          {/* Upload Button */}
          <div className="mt-8 text-center">
            <button
              onClick={handleUpload}
              disabled={!selectedFile || isUploading}
              className={`px-8 py-3 rounded-lg font-medium transition-all duration-200 ${
                selectedFile && !isUploading
                  ? 'bg-blue-600 dark:bg-blue-500 text-white hover:bg-blue-700 dark:hover:bg-blue-600 shadow-lg hover:shadow-xl'
                  : 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
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
                'Upload File'
              )}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
