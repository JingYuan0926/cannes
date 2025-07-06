import { useState, useEffect } from 'react';
import { useAccount, useWriteContract, useReadContract, useWaitForTransactionReceipt } from 'wagmi';
import { parseEther, formatEther } from 'viem';

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
    return new Date(Number(timestamp) * 1000).toLocaleString();
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-100 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h1 className="text-4xl font-bold text-center text-gray-800 mb-8">
            Subscription Management
          </h1>
          
          {!isConnected ? (
            <div className="text-center p-8 bg-gray-50 rounded-xl">
              <p className="text-gray-600 text-lg">
                Please connect your wallet to manage your subscription
              </p>
            </div>
          ) : (
            <>
              {/* Subscription Status */}
              <div className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-700 mb-4 pb-2 border-b-2 border-gray-200">
                  Subscription Status
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-gray-50 p-4 rounded-lg flex justify-between items-center">
                    <span className="font-semibold text-gray-600">Status:</span>
                    <span className={`font-bold ${isActive ? 'text-green-600' : 'text-red-600'}`}>
                      {isActive ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg flex justify-between items-center">
                    <span className="font-semibold text-gray-600">Monthly Price:</span>
                    <span className="font-medium text-gray-800">
                      {monthlyPrice ? `${formatEther(monthlyPrice)} TEST` : 'Loading...'}
                    </span>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg flex justify-between items-center">
                    <span className="font-semibold text-gray-600">Remaining Days:</span>
                    <span className="font-medium text-gray-800">
                      {remainingDays ? remainingDays.toString() : '0'} days
                    </span>
                  </div>
                </div>
              </div>

              {/* Subscription Details */}
              {subscriptionData && subscriptionData.subscriptionDate > 0 && (
                <div className="mb-8">
                  <h2 className="text-2xl font-semibold text-gray-700 mb-4 pb-2 border-b-2 border-gray-200">
                    Subscription Details
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <span className="font-semibold text-gray-600 block mb-2">Subscription Date:</span>
                      <span className="text-gray-800">{formatDate(subscriptionData.subscriptionDate)}</span>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <span className="font-semibold text-gray-600 block mb-2">Expiry Date:</span>
                      <span className="text-gray-800">{formatDate(subscriptionData.expiryDate)}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-700 mb-4 pb-2 border-b-2 border-gray-200">
                  Actions
                </h2>
                <div className="flex flex-col sm:flex-row gap-4">
                  <button
                    className="flex-1 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-300 transform hover:-translate-y-1 hover:shadow-lg disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none disabled:hover:shadow-none"
                    onClick={handleSubscribe}
                    disabled={loading || isSubscribeLoading}
                  >
                    {loading || isSubscribeLoading ? (
                      <div className="flex items-center justify-center">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                        Processing...
                      </div>
                    ) : (
                      isActive ? 'Renew Subscription' : 'Subscribe Now'
                    )}
                  </button>
                  
                  {isActive && (
                    <button
                      className="flex-1 bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-300 transform hover:-translate-y-1 hover:shadow-lg disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none disabled:hover:shadow-none"
                      onClick={handleCancel}
                      disabled={loading || isCancelLoading}
                    >
                      {loading || isCancelLoading ? (
                        <div className="flex items-center justify-center">
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                          Processing...
                        </div>
                      ) : (
                        'Cancel Subscription'
                      )}
                    </button>
                  )}
                </div>
              </div>

              {/* Transaction Status */}
              {txStatus && (
                <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-blue-800 font-medium flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                    {txStatus}
                  </p>
                </div>
              )}

              {/* Messages */}
              {error && (
                <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-800 font-medium">{error}</p>
                </div>
              )}
              
              {success && (
                <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-green-800 font-medium">{success}</p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Subscribe;
