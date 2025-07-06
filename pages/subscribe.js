import { useState, useEffect } from 'react';
import { useAccount, useWriteContract, useReadContract, useWaitForTransactionReceipt } from 'wagmi';
import { parseEther, formatEther } from 'viem';
import { motion } from 'framer-motion';
import Link from 'next/link';
import WalletConnect from '../components/WalletConnect';

// Contract ABI (simplified for the functions we need)
const SUBSCRIPTION_ABI = [
  {
    "inputs": [],
    "name": "subscribe",
    "outputs": [],
    "stateMutability": "payable",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "cancelSubscription",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [{"internalType": "address", "name": "user", "type": "address"}],
    "name": "isSubscriptionActive",
    "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [{"internalType": "address", "name": "user", "type": "address"}],
    "name": "getSubscription",
    "outputs": [
      {"internalType": "uint256", "name": "subscriptionDate", "type": "uint256"},
      {"internalType": "uint256", "name": "expiryDate", "type": "uint256"},
      {"internalType": "uint256", "name": "monthlyPrice", "type": "uint256"},
      {"internalType": "bool", "name": "isActive", "type": "bool"}
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [{"internalType": "address", "name": "user", "type": "address"}],
    "name": "getRemainingDays",
    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "monthlyPrice",
    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
    "stateMutability": "view",
    "type": "function"
  }
];

// Replace with your deployed contract address
const CONTRACT_ADDRESS = "0x863Ec8506C15D056F43d9BBA811ccB819c3DDFE9";

const Subscribe = () => {
  const { address, isConnected } = useAccount();
  const [subscriptionData, setSubscriptionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [txStatus, setTxStatus] = useState('');

  // Contract write functions
  const { writeContract: subscribe, data: subscribeHash } = useWriteContract();
  const { writeContract: cancelSubscription, data: cancelHash } = useWriteContract();

  // Contract read functions
  const { data: monthlyPrice, refetch: refetchPrice } = useReadContract({
    address: CONTRACT_ADDRESS,
    abi: SUBSCRIPTION_ABI,
    functionName: 'monthlyPrice',
  });

  const { data: isActive, refetch: refetchIsActive } = useReadContract({
    address: CONTRACT_ADDRESS,
    abi: SUBSCRIPTION_ABI,
    functionName: 'isSubscriptionActive',
    args: [address],
    enabled: !!address,
  });

  const { data: subscription, refetch: refetchSubscription } = useReadContract({
    address: CONTRACT_ADDRESS,
    abi: SUBSCRIPTION_ABI,
    functionName: 'getSubscription',
    args: [address],
    enabled: !!address,
  });

  const { data: remainingDays, refetch: refetchRemainingDays } = useReadContract({
    address: CONTRACT_ADDRESS,
    abi: SUBSCRIPTION_ABI,
    functionName: 'getRemainingDays',
    args: [address],
    enabled: !!address,
  });

  // Wait for transaction receipts
  const { isLoading: isSubscribeLoading, isSuccess: isSubscribeSuccess } = useWaitForTransactionReceipt({
    hash: subscribeHash,
  });

  const { isLoading: isCancelLoading, isSuccess: isCancelSuccess } = useWaitForTransactionReceipt({
    hash: cancelHash,
  });

  // Handle successful transactions
  useEffect(() => {
    if (isSubscribeSuccess && subscribeHash) {
      setSuccess('Subscription successful! Your subscription is now active.');
      setTxStatus('');
      setLoading(false);
      // Refetch all data after successful transaction
      refetchIsActive();
      refetchSubscription();
      refetchRemainingDays();
    }
  }, [isSubscribeSuccess, subscribeHash, refetchIsActive, refetchSubscription, refetchRemainingDays]);

  useEffect(() => {
    if (isCancelSuccess && cancelHash) {
      setSuccess('Subscription cancelled successfully!');
      setTxStatus('');
      setLoading(false);
      // Refetch all data after successful transaction
      refetchIsActive();
      refetchSubscription();
      refetchRemainingDays();
    }
  }, [isCancelSuccess, cancelHash, refetchIsActive, refetchSubscription, refetchRemainingDays]);

  // Handle transaction loading states
  useEffect(() => {
    if (isSubscribeLoading) {
      setTxStatus('Waiting for transaction confirmation...');
    }
  }, [isSubscribeLoading]);

  useEffect(() => {
    if (isCancelLoading) {
      setTxStatus('Waiting for transaction confirmation...');
    }
  }, [isCancelLoading]);

  // Handle subscription
  const handleSubscribe = async () => {
    if (!isConnected) {
      setError('Please connect your wallet first');
      return;
    }

    try {
      setLoading(true);
      setError('');
      setSuccess('');
      setTxStatus('Please confirm transaction in your wallet...');

      await subscribe({
        address: CONTRACT_ADDRESS,
        abi: SUBSCRIPTION_ABI,
        functionName: 'subscribe',
        value: monthlyPrice,
      });

      setTxStatus('Transaction submitted! Waiting for confirmation...');
    } catch (err) {
      setError(`Subscription failed: ${err.message}`);
      setTxStatus('');
      setLoading(false);
    }
  };

  // Handle cancel subscription
  const handleCancel = async () => {
    if (!isConnected) {
      setError('Please connect your wallet first');
      return;
    }

    try {
      setLoading(true);
      setError('');
      setSuccess('');
      setTxStatus('Please confirm transaction in your wallet...');

      await cancelSubscription({
        address: CONTRACT_ADDRESS,
        abi: SUBSCRIPTION_ABI,
        functionName: 'cancelSubscription',
      });

      setTxStatus('Transaction submitted! Waiting for confirmation...');
    } catch (err) {
      setError(`Cancellation failed: ${err.message}`);
      setTxStatus('');
      setLoading(false);
    }
  };

  // Format date helper
  const formatDate = (timestamp) => {
    const date = new Date(Number(timestamp) * 1000);
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = String(date.getFullYear()).slice(-2);
    return `${day}${month}${year}`;
  };

  // Update subscription data when contract data changes
  useEffect(() => {
    if (subscription && address) {
      setSubscriptionData({
        subscriptionDate: subscription[0],
        expiryDate: subscription[1],
        monthlyPrice: subscription[2],
        isActive: subscription[3],
      });
    }
  }, [subscription, address]);

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
        <div className="flex bg-white/80 backdrop-blur-sm border border-blue-200 rounded-full p-1 transition-all duration-300 shadow-lg hover:shadow-xl">
          <Link href="/analyse">
            <div className="px-6 py-2 rounded-full hover:bg-blue-100 text-slate-800 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Analyse
            </div>
          </Link>
          <Link href="/upload">
            <div className="px-6 py-2 rounded-full hover:bg-blue-100 text-slate-800 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              Upload
            </div>
          </Link>
          <Link href="/view">
            <div className="px-6 py-2 rounded-full hover:bg-blue-100 text-slate-800 font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95">
              View
            </div>
          </Link>
          <Link href="/subscribe">
            <div className="px-6 py-2 rounded-full bg-blue-600 text-white font-medium text-sm transition-all duration-300 cursor-pointer transform hover:scale-105 active:scale-95 shadow-md">
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
        <h1 className="text-3xl font-bold text-center text-slate-900 transform transition-all duration-300">Subscription Management</h1>
        <p className="text-center text-slate-600 mt-2 transition-opacity duration-200">
          Manage your subscription and access premium features
        </p>
      </motion.div>

      {/* Main Content */}
      <motion.main 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="flex-1 flex flex-col items-center px-8 min-h-0 overflow-y-auto justify-start pt-8"
      >
        <div className="max-w-4xl w-full">
          
          {!isConnected ? (
            <motion.div 
              variants={itemVariants}
              className="text-center p-8 bg-white/80 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg"
            >
              <svg className="w-16 h-16 text-blue-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
              <h3 className="text-xl font-semibold text-slate-900 mb-2">Wallet Connection Required</h3>
              <p className="text-slate-600 text-lg">
                Please connect your wallet to manage your subscription
              </p>
            </motion.div>
          ) : (
            <>
              {/* Subscription Status Cards */}
              <motion.div 
                variants={itemVariants}
                className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8"
              >
                <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-blue-200 transition-all duration-300 shadow-lg hover:shadow-xl">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-medium text-slate-600 mb-1">Status</h3>
                      <p className={`text-2xl font-bold ${isActive ? 'text-emerald-600' : 'text-rose-600'}`}>
                        {isActive ? 'Active' : 'Inactive'}
                      </p>
                    </div>
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                      isActive ? 'bg-emerald-100' : 'bg-rose-100'
                    }`}>
                      {isActive ? (
                        <svg className="w-6 h-6 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      ) : (
                        <svg className="w-6 h-6 text-rose-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-blue-200 transition-all duration-300 shadow-lg hover:shadow-xl">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-medium text-slate-600 mb-1">Monthly Price</h3>
                      <p className="text-2xl font-bold text-slate-900">
                        0.01 TEST
                      </p>
                    </div>
                    <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                      <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
                      </svg>
                    </div>
                  </div>
                </div>
                
                <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-blue-200 transition-all duration-300 shadow-lg hover:shadow-xl">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-medium text-slate-600 mb-1">Remaining Days</h3>
                      <p className="text-2xl font-bold text-slate-900">
                        {remainingDays ? remainingDays.toString() : '0'}
                      </p>
                    </div>
                    <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
                      <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Subscription Details */}
              {subscriptionData && subscriptionData.subscriptionDate > 0 && (
                <motion.div 
                  variants={itemVariants}
                  className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-blue-200 transition-all duration-300 shadow-lg hover:shadow-xl mb-8"
                >
                  <h3 className="text-lg font-semibold text-slate-900 mb-4">Subscription Details</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-blue-50/80 backdrop-blur-sm p-4 rounded-lg border border-blue-100">
                      <span className="font-medium text-slate-600 block mb-2">Subscription Date</span>
                      <span className="text-slate-900">{formatDate(subscriptionData.subscriptionDate)}</span>
                    </div>
                    <div className="bg-blue-50/80 backdrop-blur-sm p-4 rounded-lg border border-blue-100">
                      <span className="font-medium text-slate-600 block mb-2">Expiry Date</span>
                      <span className="text-slate-900">{formatDate(subscriptionData.expiryDate)}</span>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Action Buttons */}
              <motion.div 
                variants={itemVariants}
                className="flex flex-col sm:flex-row gap-4 mb-8"
              >
                <button
                  onClick={handleSubscribe}
                  disabled={loading || isSubscribeLoading}
                  className={`flex-1 py-3 px-6 rounded-lg font-medium transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl ${
                    loading || isSubscribeLoading
                      ? 'bg-slate-300 text-slate-500 cursor-not-allowed scale-100'
                      : 'bg-blue-600 text-white hover:bg-blue-700 hover:-translate-y-1'
                  }`}
                >
                  {loading || isSubscribeLoading ? (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-2"></div>
                      Processing...
                    </div>
                  ) : (
                    isActive ? 'Renew Subscription' : 'Subscribe Now'
                  )}
                </button>
                
                {isActive && (
                  <button
                    onClick={handleCancel}
                    disabled={loading || isCancelLoading}
                    className={`flex-1 py-3 px-6 rounded-lg font-medium transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl ${
                      loading || isCancelLoading
                        ? 'bg-slate-300 text-slate-500 cursor-not-allowed scale-100'
                        : 'bg-rose-600 text-white hover:bg-rose-700 hover:-translate-y-1'
                    }`}
                  >
                    {loading || isCancelLoading ? (
                      <div className="flex items-center justify-center">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-rose-600 mr-2"></div>
                        Processing...
                      </div>
                    ) : (
                      'Cancel Subscription'
                    )}
                  </button>
                )}
              </motion.div>

              {/* Status Messages */}
              {txStatus && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                  className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-2xl shadow-lg"
                >
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
                    <p className="text-blue-800 font-medium">{txStatus}</p>
                  </div>
                </motion.div>
              )}

              {error && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                  className="mb-6 p-4 bg-red-50 border border-red-200 rounded-2xl shadow-lg"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <svg className="w-6 h-6 text-red-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <p className="text-red-800 font-medium">{error}</p>
                    </div>
                    <button
                      onClick={() => setError('')}
                      className="text-red-500 hover:text-red-700 ml-2"
                    >
                      ×
                    </button>
                  </div>
                </motion.div>
              )}
              
              {success && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                  className="mb-6 p-4 bg-green-50 border border-green-200 rounded-2xl shadow-lg"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <svg className="w-6 h-6 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <p className="text-green-800 font-medium">{success}</p>
                    </div>
                    <button
                      onClick={() => setSuccess('')}
                      className="text-green-500 hover:text-green-700 ml-2"
                    >
                      ×
                    </button>
                  </div>
                </motion.div>
              )}
            </>
          )}
        </div>
      </motion.main>
    </div>
  );
};

export default Subscribe;
