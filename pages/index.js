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
              Chat
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
            <motion.div 
              variants={itemVariants}
              className="text-center p-8 bg-white/80 backdrop-blur-sm rounded-2xl border border-blue-200 shadow-lg"
            >
              <svg className="w-16 h-16 text-blue-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              <h3 className="text-xl font-semibold text-slate-900 mb-2">Please Subscribe to Use Our Service</h3>
              <p className="text-slate-600 text-lg mb-6">
                Get access to premium data analysis features with our subscription plan
              </p>
              <Link href="/subscribe">
                <button className="py-3 px-6 bg-blue-600 text-white rounded-lg font-medium transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl hover:bg-blue-700 hover:-translate-y-1">
                  Subscribe Now
                </button>
              </Link>
            </motion.div>
          )}
        </div>
      </motion.main>
    </div>
  );
};

export default Subscribe;
