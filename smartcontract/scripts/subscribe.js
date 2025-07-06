const { ethers } = require("hardhat");

async function main() {
  // Get the contract factory and deploy
  console.log("Deploying Subscription contract...");
  const Subscription = await ethers.getContractFactory("Subscription");
  const subscription = await Subscription.deploy();
  await subscription.waitForDeployment();
  
  const contractAddress = await subscription.getAddress();
  console.log(`Subscription contract deployed to: ${contractAddress}`);
  
  // Get signers
  const [owner, user1, user2] = await ethers.getSigners();
  console.log(`Owner: ${owner.address}`);
  console.log(`User1: ${user1.address}`);
  console.log(`User2: ${user2.address}`);
  
  // Get monthly price
  const monthlyPrice = await subscription.monthlyPrice();
  console.log(`Monthly price: ${ethers.formatEther(monthlyPrice)} TEST tokens`);
  
  console.log("\n--- Testing Subscription System ---");
  
  // User1 subscribes
  console.log("\n1. User1 subscribing...");
  const tx1 = await subscription.connect(user1).subscribe({ value: monthlyPrice });
  await tx1.wait();
  console.log("✅ User1 subscribed successfully!");
  
  // Check User1's subscription
  const user1Sub = await subscription.getSubscription(user1.address);
  const user1Active = await subscription.isSubscriptionActive(user1.address);
  const user1RemainingDays = await subscription.getRemainingDays(user1.address);
  
  console.log(`User1 subscription date: ${new Date(Number(user1Sub[0]) * 1000).toLocaleString()}`);
  console.log(`User1 expiry date: ${new Date(Number(user1Sub[1]) * 1000).toLocaleString()}`);
  console.log(`User1 monthly price: ${ethers.formatEther(user1Sub[2])} TEST`);
  console.log(`User1 is active: ${user1Active}`);
  console.log(`User1 remaining days: ${user1RemainingDays}`);
  
  // User2 subscribes
  console.log("\n2. User2 subscribing...");
  const tx2 = await subscription.connect(user2).subscribe({ value: monthlyPrice });
  await tx2.wait();
  console.log("✅ User2 subscribed successfully!");
  
  // Check User2's subscription
  const user2Sub = await subscription.getSubscription(user2.address);
  const user2Active = await subscription.isSubscriptionActive(user2.address);
  const user2RemainingDays = await subscription.getRemainingDays(user2.address);
  
  console.log(`User2 subscription date: ${new Date(Number(user2Sub[0]) * 1000).toLocaleString()}`);
  console.log(`User2 expiry date: ${new Date(Number(user2Sub[1]) * 1000).toLocaleString()}`);
  console.log(`User2 monthly price: ${ethers.formatEther(user2Sub[2])} TEST`);
  console.log(`User2 is active: ${user2Active}`);
  console.log(`User2 remaining days: ${user2RemainingDays}`);
  
  // User1 renews subscription
  console.log("\n3. User1 renewing subscription...");
  const tx3 = await subscription.connect(user1).subscribe({ value: monthlyPrice });
  await tx3.wait();
  console.log("✅ User1 renewed subscription!");
  
  // Check User1's updated subscription
  const user1SubRenewed = await subscription.getSubscription(user1.address);
  const user1RemainingDaysRenewed = await subscription.getRemainingDays(user1.address);
  
  console.log(`User1 new expiry date: ${new Date(Number(user1SubRenewed[1]) * 1000).toLocaleString()}`);
  console.log(`User1 remaining days after renewal: ${user1RemainingDaysRenewed}`);
  
  // User2 cancels subscription
  console.log("\n4. User2 cancelling subscription...");
  const tx4 = await subscription.connect(user2).cancelSubscription();
  await tx4.wait();
  console.log("✅ User2 cancelled subscription!");
  
  // Check User2's subscription after cancellation
  const user2ActiveAfterCancel = await subscription.isSubscriptionActive(user2.address);
  console.log(`User2 is active after cancellation: ${user2ActiveAfterCancel}`);
  
  // Check contract balance
  const contractBalance = await subscription.getContractBalance();
  console.log(`\nContract balance: ${ethers.formatEther(contractBalance)} TEST tokens`);
  
  console.log("\n--- Subscription System Test Complete ---");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 