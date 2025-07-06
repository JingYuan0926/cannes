// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

contract Subscription {
    struct UserSubscription {
        address user;
        uint256 subscriptionDate;
        uint256 expiryDate;
        uint256 monthlyPrice;
        bool isActive;
    }

    event SubscriptionCreated(
        address indexed user,
        uint256 subscriptionDate,
        uint256 expiryDate,
        uint256 monthlyPrice
    );

    event SubscriptionRenewed(
        address indexed user,
        uint256 newExpiryDate,
        uint256 monthlyPrice
    );

    event SubscriptionCancelled(address indexed user);

    mapping(address => UserSubscription) public subscriptions;
    uint256 public monthlyPrice = 1 ether; // 1 TEST token per month
    address public owner;

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function subscribe() external payable {
        require(msg.value >= monthlyPrice, "Insufficient payment");
        
        UserSubscription storage userSub = subscriptions[msg.sender];
        
        if (userSub.user == address(0)) {
            // New subscription
            userSub.user = msg.sender;
            userSub.subscriptionDate = block.timestamp;
            userSub.expiryDate = block.timestamp + 30 days;
            userSub.monthlyPrice = monthlyPrice;
            userSub.isActive = true;
            
            emit SubscriptionCreated(
                msg.sender,
                userSub.subscriptionDate,
                userSub.expiryDate,
                monthlyPrice
            );
        } else {
            // Renew existing subscription
            if (userSub.expiryDate > block.timestamp) {
                // Extend from current expiry
                userSub.expiryDate += 30 days;
            } else {
                // Extend from now if expired
                userSub.expiryDate = block.timestamp + 30 days;
            }
            userSub.isActive = true;
            
            emit SubscriptionRenewed(
                msg.sender,
                userSub.expiryDate,
                monthlyPrice
            );
        }
    }

    function cancelSubscription() external {
        UserSubscription storage userSub = subscriptions[msg.sender];
        require(userSub.user != address(0), "No subscription found");
        
        userSub.isActive = false;
        emit SubscriptionCancelled(msg.sender);
    }

    function isSubscriptionActive(address user) external view returns (bool) {
        UserSubscription memory userSub = subscriptions[user];
        return userSub.isActive && userSub.expiryDate > block.timestamp;
    }

    function getSubscription(address user) external view returns (
        uint256 subscriptionDate,
        uint256 expiryDate,
        uint256 monthlyPrice,
        bool isActive
    ) {
        UserSubscription memory userSub = subscriptions[user];
        return (
            userSub.subscriptionDate,
            userSub.expiryDate,
            userSub.monthlyPrice,
            userSub.isActive
        );
    }

    function getRemainingDays(address user) external view returns (uint256) {
        UserSubscription memory userSub = subscriptions[user];
        if (userSub.expiryDate <= block.timestamp) {
            return 0;
        }
        return (userSub.expiryDate - block.timestamp) / 1 days;
    }

    function setMonthlyPrice(uint256 newPrice) external onlyOwner {
        monthlyPrice = newPrice;
    }

    function withdrawFunds() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    function getContractBalance() external view returns (uint256) {
        return address(this).balance;
    }
} 