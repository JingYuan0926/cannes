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
  const [isUploadComplete, setIsUploadComplete] = useState(false);

  // Dark mode persistence
  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedDarkMode !== null) {
      setDarkMode(JSON.parse(savedDarkMode));
    } else {
      setDarkMode(false); // Default to light mode instead of system preference
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
      setIsUploadComplete(false);
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
      setIsUploadComplete(false);
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      setIsUploading(true);
      setUploadProgress(0);
      setIsUploadComplete(false);
      
      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setIsUploading(false);
            setIsUploadComplete(true);
            // Show success message
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
    <div className={`${geistSans.className} ${geistMono.className} min-h-screen font-[family-name:var(--font-geist-sans)] bg-white text-gray-900 transition-colors duration-300`}>
      {/* Navigation Bar */}
      <nav className="flex justify-center pt-8 pb-4">
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
        </div>
      </nav>

      {/* Page Header */}
      <div className="px-8 py-4">
        <h1 className="text-3xl font-bold text-center text-black transform transition-all duration-300">Upload Your Data</h1>
        <p className="text-center text-gray-600 mt-2 transition-opacity duration-200">
          Upload your datasets to start analyzing and discovering insights
        </p>
      </div>

      {/* Main Content */}
      <main className="flex flex-col items-center justify-center min-h-[calc(100vh-200px)] px-8">
        <div className="max-w-2xl w-full">
          
          {/* File Upload Area */}
          <div
            className={`border-2 rounded-2xl p-12 text-center transition-all duration-300 transform hover:scale-[1.02] ${
              selectedFile
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
                {selectedFile ? (
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
                {selectedFile ? (
                  <div className="animate-fadeIn">
                    <p className="text-lg font-medium mb-2 text-green-700">
                      ✓ {selectedFile.name}
                    </p>
                    <p className="text-sm text-black">
                      Ready to upload
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className="text-lg font-medium mb-2 text-black transition-all duration-200">
                      Drop your file here
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
              />
              
              {!selectedFile && (
                <label
                  htmlFor="fileInput"
                  className="inline-block px-6 py-3 bg-gray-300 text-black rounded-lg font-medium cursor-pointer hover:bg-gray-400 transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-md hover:shadow-lg"
                >
                  Select File
                </label>
              )}

              {selectedFile && (
                <div className="flex gap-2 justify-center animate-slideInUp">
                  <label
                    htmlFor="fileInput"
                    className="px-4 py-2 bg-gray-300 text-black rounded-lg font-medium cursor-pointer hover:bg-gray-400 transition-all duration-200 text-sm transform hover:scale-105 active:scale-95 shadow-sm hover:shadow-md"
                  >
                    Change File
                  </label>
                  <button
                    onClick={() => {
                      setSelectedFile(null);
                      setIsUploadComplete(false);
                    }}
                    className="px-4 py-2 bg-red-100 text-red-700 rounded-lg font-medium hover:bg-red-200 transition-all duration-200 text-sm transform hover:scale-105 active:scale-95 shadow-sm hover:shadow-md"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Supported file types */}
          <div className="mt-4 text-center">
            <p className="text-sm text-gray-600 transition-opacity duration-200">
              Supported formats: CSV, Excel (.xlsx, .xls), JSON, TXT
            </p>
          </div>
          
          {/* File Information */}
          {selectedFile && (
            <div className="mt-6 bg-gray-200 p-6 rounded-2xl border border-gray-300 transition-all duration-300 shadow-md hover:shadow-lg animate-slideInUp">
              <h3 className="font-semibold text-lg mb-4 text-black transform transition-all duration-200">File Details</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-gray-300 rounded-lg transition-all duration-200">
                  <p className="text-sm text-black">Name</p>
                  <p className="font-medium text-black truncate">{selectedFile.name}</p>
                </div>
                <div className="text-center p-4 bg-gray-300 rounded-lg transition-all duration-200">
                  <p className="text-sm text-black">Size</p>
                  <p className="font-medium text-black">{formatFileSize(selectedFile.size)}</p>
                </div>
                <div className="text-center p-4 bg-gray-300 rounded-lg transition-all duration-200">
                  <p className="text-sm text-black">Type</p>
                  <p className="font-medium text-black">{selectedFile.type || 'Unknown'}</p>
                </div>
              </div>
            </div>
          )}

          {/* Upload Progress */}
          {isUploading && (
            <div className="mt-6 bg-gray-200 p-6 rounded-2xl border border-gray-300 animate-slideInUp shadow-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-black">Uploading...</span>
                <span className="text-sm text-black animate-pulse">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-300 rounded-full h-2 overflow-hidden">
                <div 
                  className="bg-gray-600 h-2 rounded-full transition-all duration-300 ease-out animate-pulse"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          )}

          {/* Success Message */}
          {isUploadComplete && (
            <div className="mt-6 bg-green-50 p-6 rounded-2xl border border-green-200 animate-slideInUp shadow-lg">
              <div className="flex items-center justify-center">
                <svg className="w-6 h-6 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-green-700 font-medium">Upload completed successfully!</span>
              </div>
            </div>
          )}
          
          {/* Upload Button */}
          <div className="mt-8 text-center">
            <button
              onClick={handleUpload}
              disabled={!selectedFile || isUploading}
              className={`px-8 py-3 rounded-lg font-medium transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl ${
                selectedFile && !isUploading
                  ? 'bg-gray-600 text-white hover:bg-gray-700 hover:-translate-y-1'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed scale-100'
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

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes slideInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }

        .animate-slideInUp {
          animation: slideInUp 0.4s ease-out;
        }
      `}</style>
    </div>
  );
}
